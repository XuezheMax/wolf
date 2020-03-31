import os
import sys
import gc

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import random
import math
import numpy as np

import torch
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image


from wolf.data import load_datasets, get_batch, preprocess, postprocess
from wolf import WolfModel
from wolf.utils import total_grad_norm
from wolf.optim import ExponentialScheduler

from experiments.options import parse_args


def is_master(rank):
    return rank <= 0


def is_distributed(rank):
    return rank >= 0


def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()


def get_optimizer(learning_rate, parameters, betas, eps, amsgrad, step_decay, weight_decay, warmup_steps, init_lr):
    optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
    step_decay = step_decay
    scheduler = ExponentialScheduler(optimizer, step_decay, warmup_steps, init_lr)
    return optimizer, scheduler


def setup(args):
    def check_dataset():
        if dataset == 'cifar10':
            assert image_size == 32, 'CIFAR-10 expected image size 32 but got {}'.format(image_size)
        elif dataset.startswith('lsun'):
            assert image_size in [128, 256]
        elif dataset == 'celeba':
            assert image_size in [256, 512]
        elif dataset == 'imagenet':
            assert image_size in [64, 128, 256]

    dataset = args.dataset
    if args.category is not None:
        dataset = dataset + '_' + args.category
    image_size = args.image_size
    check_dataset()

    nc = 3
    args.nx = image_size ** 2 * nc
    n_bits = args.n_bits
    args.n_bins = 2. ** n_bits
    args.test_k = 5

    model_path = args.model_path
    args.checkpoint_name = os.path.join(model_path, 'checkpoint')

    result_path = os.path.join(model_path, 'images')
    args.result_path = result_path
    data_path = args.data_path

    if is_master(args.rank):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if args.recover < 0:
            args.log = open(os.path.join(model_path, 'log.txt'), 'w')
        else:
            args.log = open(os.path.join(model_path, 'log.txt'), 'a')
    else:
        args.log = None

    args.cuda = torch.cuda.is_available()
    random_seed = args.seed + args.rank if args.rank >= 0 else args.seed
    if args.recover >= 0:
        random_seed += random.randint(0, 1024)
    logging("Rank {}: random seed={}".format(args.rank, random_seed), logfile=args.log)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device('cuda', args.local_rank) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = True

    args.world_size = int(os.environ["WORLD_SIZE"]) if is_distributed(args.rank) else 1
    logging("Rank {}: ".format(args.rank) + str(args), args.log)

    train_data, val_data = load_datasets(dataset, image_size, data_path=data_path)
    train_index = np.arange(len(train_data))
    np.random.shuffle(train_index)
    val_index = np.arange(len(val_data))

    if is_master(args.rank):
        logging('Data size: training: {}, val: {}'.format(len(train_index), len(val_index)))

    if args.recover >= 0:
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    else:
        params = json.load(open(args.config, 'r'))
        json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)

    wolf = WolfModel.from_params(params)
    wolf.to_device(device)
    args.device = device

    return args, (train_data, val_data), (train_index, val_index), wolf


def init_dataloader(args, train_data, val_data):
    if is_distributed(args.rank):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=args.rank,
                                                                        num_replicas=args.world_size,
                                                                        shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    if is_master(args.rank):
        eval_batch = args.eval_batch_size
        val_loader = DataLoader(val_data, batch_size=eval_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    else:
        val_loader = None

    return train_loader, train_sampler, val_loader


def init_model(args, train_data, train_index, wolf):
    wolf.eval()
    init_batch_size = args.init_batch_size
    logging('Rank {}, init model: {} instances'.format(args.rank, init_batch_size), args.log)
    init_index = np.random.choice(train_index, init_batch_size, replace=False)
    init_x, init_y = get_batch(train_data, init_index)
    init_x = preprocess(init_x.to(args.device), args.n_bits)
    init_y = init_y.to(args.device)
    wolf.init(init_x, y=init_y, init_scale=1.0)


def reconstruct(args, epoch, val_data, val_index, wolf):
    logging('reconstruct', args.log)
    wolf.eval()
    n = 16
    np.random.shuffle(val_index)
    img, y = get_batch(val_data, val_index[:n])
    img = img.to(args.device)
    y = y.to(args.device)

    z, epsilon = wolf.encode(img, y=y, n_bits=args.n_bits, random=False)
    epsilon = epsilon.squeeze(1)
    z = z.squeeze(1) if z is not None else z
    img_recon = wolf.decode(epsilon, z=z, n_bits=args.n_bits)

    img = postprocess(preprocess(img, args.n_bits), args.n_bits)
    abs_err = img_recon.add(img * -1).abs()
    logging('Err: {:.4f}, {:.4f}'.format(abs_err.max().item(), abs_err.mean().item()), args.log)

    comparison = torch.cat([img, img_recon], dim=0).cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct{}.png'.format(epoch)
    save_image(comparison, os.path.join(args.result_path, image_file), nrow=16)


def sample(args, epoch, wolf):
    logging('sampling', args.log)
    wolf.eval()
    n = 64 if args.image_size > 128 else 256
    nrow = int(math.sqrt(n))
    taus = [0.7, 0.8, 0.9, 1.0]
    start_time = time.time()
    image_size = (3, args.image_size, args.image_size)
    for t in taus:
        imgs = wolf.synthesize(n, image_size, tau=t, n_bits=args.n_bits, device=args.device)
        image_file = 'sample{}.t{:.1f}.png'.format(epoch, t)
        save_image(imgs, os.path.join(args.result_path, image_file), nrow=nrow)
    logging('time: {:.1f}s'.format(time.time() - start_time), args.log)


def eval(args, val_loader, wolf):
    wolf.eval()
    wolf.sync()
    gnll = 0
    nent = 0
    kl = 0
    num_insts = 0
    device = args.device
    n_bits = args.n_bits
    n_bins = args.n_bins
    nx = args.nx
    test_k = args.test_k
    for data, y in val_loader:
        batch_size = len(data)
        data = data.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_gen, loss_kl, loss_dequant = wolf.loss(data, y=y, n_bits=n_bits, nsamples=test_k)
        gnll += loss_gen.sum().item()
        kl += loss_kl.sum().item()
        nent += loss_dequant.sum().item()
        num_insts += batch_size

    gnll = gnll / num_insts
    nent = nent / num_insts
    kl = kl / num_insts
    nll = gnll + kl + nent + np.log(n_bins / 2.) * nx
    bpd = nll / (nx * np.log(2.0))
    nepd = nent / (nx * np.log(2.0))
    logging('Avg  NLL: {:.2f}, KL: {:.2f}, NENT: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}'.format(
        nll, kl, nent, bpd, nepd), args.log)
    return nll, kl, nent, bpd, nepd


def train(args, train_loader, train_index, train_sampler, val_loader, val_data, val_index, wolf):
    epochs = args.epochs
    train_k = args.train_k
    n_bits = args.n_bits
    n_bins = args.n_bins
    nx = args.nx
    grad_clip = args.grad_clip
    batch_steps = args.batch_steps

    steps_per_checkpoint = 1000

    device = args.device
    log = args.log

    lr_warmups = args.warmup_steps
    init_lr = 1e-7
    betas = (args.beta1, args.beta2)
    eps = args.eps
    amsgrad = args.amsgrad
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay

    optimizer, scheduler = get_optimizer(args.lr, wolf.parameters(), betas, eps,
                                         amsgrad=amsgrad, step_decay=lr_decay,
                                         weight_decay=weight_decay,
                                         warmup_steps=lr_warmups, init_lr=init_lr)
    if args.recover >= 0:
        checkpoint_name = args.checkpoint_name + '{}.tar'.format(args.recover)
        print(f"Rank = {args.rank}, loading from checkpoint {checkpoint_name}")

        checkpoint = torch.load(checkpoint_name, map_location=args.device)
        start_epoch = checkpoint['epoch']
        last_step = checkpoint['step']
        wolf.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        best_epoch = checkpoint['best_epoch']
        best_nll = checkpoint['best_nll']
        best_bpd = checkpoint['best_bpd']
        best_nent = checkpoint['best_nent']
        best_nepd = checkpoint['best_nepd']
        best_kl = checkpoint['best_kl']
        del checkpoint
        if is_master(args.rank):
            with torch.no_grad():
                logging('Evaluating after resuming model...', log)
                eval(args, val_loader, wolf)
    else:
        start_epoch = 1
        last_step = -1
        best_epoch = 0
        best_nll = 1e12
        best_bpd = 1e12
        best_nent = 1e12
        best_nepd = 1e12
        best_kl = 1e12

    for epoch in range(start_epoch, epochs + 1):
        wolf.train()
        if is_distributed(args.rank):
            train_sampler.set_epoch(epoch)

        lr = scheduler.get_lr()[0]
        start_time = time.time()
        if is_master(args.rank):
            logging('Epoch: %d (lr=%.6f, betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s, lr decay=%.6f, clip=%.1f, l2=%.1e, train_k=%d)' % (
            epoch, lr, betas[0], betas[1], eps, amsgrad, lr_decay, grad_clip, weight_decay, train_k), log)

        gnll = torch.Tensor([0.]).to(device)
        kl = torch.Tensor([0.]).to(device)
        nent = torch.Tensor([0.]).to(device)
        num_insts = torch.Tensor([0.]).to(device)
        num_back = 0
        num_nans = 0
        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()
        for step, (data, y) in enumerate(train_loader):
            if step <= last_step:
                continue
            last_step = -1
            optimizer.zero_grad()
            batch_size = len(data)
            data = data.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            data_list = [data,] if batch_steps == 1 else data.chunk(batch_steps, dim=0)
            y_list = [y,] if batch_steps == 1 else y.chunk(batch_steps, dim=0)

            gnll_batch = 0
            kl_batch = 0
            nent_batch = 0
            # disable allreduce for accumulated gradient.
            if is_distributed(args.rank):
                wolf.disable_allreduce()
            for data, y in zip (data_list[:-1], y_list[:-1]):
                loss_gen, loss_kl, loss_dequant = wolf.loss(data, y=y, n_bits=n_bits, nsamples=train_k)
                loss_gen = loss_gen.sum()
                loss_kl = loss_kl.sum()
                loss_dequant = loss_dequant.sum()
                loss = (loss_gen + loss_kl + loss_dequant) / batch_size
                loss.backward()
                with torch.no_grad():
                    gnll_batch += loss_gen.item()
                    kl_batch += loss_kl.item()
                    nent_batch += loss_dequant.item()
            # enable allreduce for the last step.
            if is_distributed(args.rank):
                wolf.enable_allreduce()
            data, y = data_list[-1], y_list[-1]
            loss_gen, loss_kl, loss_dequant = wolf.loss(data, y=y, n_bits=n_bits, nsamples=train_k)
            loss_gen = loss_gen.sum()
            loss_kl = loss_kl.sum()
            loss_dequant = loss_dequant.sum()
            loss = (loss_gen + loss_kl + loss_dequant) / batch_size
            loss.backward()
            with torch.no_grad():
                gnll_batch += loss_gen.item()
                kl_batch += loss_kl.item()
                nent_batch += loss_dequant.item()

            if grad_clip > 0:
                grad_norm = clip_grad_norm_(wolf.parameters(), grad_clip)
            else:
                grad_norm = total_grad_norm(wolf.parameters())

            if math.isnan(grad_norm):
                num_nans += 1
            else:
                optimizer.step()
                scheduler.step()
                num_insts += batch_size
                gnll += gnll_batch
                kl += kl_batch
                nent += nent_batch

            if step % 10 == 0:
                torch.cuda.empty_cache()

            if step % args.log_interval == 0 and is_master(args.rank):
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                nums = max(num_insts.item(), 1)
                train_gnll = gnll.item() / nums
                train_kl = kl.item() / nums
                train_nent = nent.item() / nums
                train_nll = train_gnll + train_kl + train_nent + np.log(n_bins / 2.) * nx
                bits_per_pixel = train_nll / (nx * np.log(2.0))
                nent_per_pixel = train_nent / (nx * np.log(2.0))
                curr_lr = scheduler.get_lr()[0]
                log_info = '[{}/{} ({:.0f}%) lr={:.6f}, {}] NLL: {:.2f}, BPD: {:.4f}, KL: {:.2f}, NENT: {:.2f}, NEPD: {:.4f}'.format(
                    step * batch_size * args.world_size, len(train_index),
                    100. * step * batch_size * args.world_size / len(train_index), curr_lr, num_nans,
                    train_nll, bits_per_pixel, train_kl, train_nent, nent_per_pixel)

                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

            if step > 0 and step % steps_per_checkpoint == 0 and is_master(args.rank):
                # save checkpoint
                checkpoint_name = args.checkpoint_name + '{}.tar'.format(step)
                torch.save({'epoch': epoch,
                            'step': step,
                            'model': wolf.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_epoch': best_epoch,
                            'best_nll': best_nll,
                            'best_bpd': best_bpd,
                            'best_kl': best_kl,
                            'best_nent': best_nent,
                            'best_nepd': best_nepd},
                           checkpoint_name)

        if is_distributed(args.rank):
            dist.reduce(gnll, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(kl, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(nent, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(num_insts, dst=0, op=dist.ReduceOp.SUM)

        if is_master(args.rank):
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            nums = num_insts.item()
            train_gnll = gnll.item() / nums
            train_kl = kl.item() / nums
            train_nent = nent.item() / nums
            train_nll = train_gnll + train_kl + train_nent + np.log(n_bins / 2.) * nx
            bits_per_pixel = train_nll / (nx * np.log(2.0))
            nent_per_pixel = train_nent / (nx * np.log(2.0))
            logging('Average NLL: {:.2f}, BPD: {:.4f}, KL: {:.2f}, NENT: {:.2f}, NEPD: {:.4f}, time: {:.1f}s'.format(
                    train_nll, bits_per_pixel, train_kl, train_nent, nent_per_pixel, time.time() - start_time), log)
            logging('-' * 125, log)

            if epoch < args.valid_epochs or epoch % args.valid_epochs == 0:
                with torch.no_grad():
                    nll, kl, nent, bpd, nepd = eval(args, val_loader, wolf)
                    if nll < best_nll:
                        best_epoch = epoch
                        best_nll = nll
                        best_bpd = bpd
                        best_kl = kl
                        best_nent = nent
                        best_nepd = nepd
                        wolf.save(args.model_path)
                        checkpoint_name = args.checkpoint_name + '{}.tar'.format(0)
                        torch.save({'epoch': epoch + 1,
                                    'step': -1,
                                    'model': wolf.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_nll': best_nll,
                                    'best_bpd': best_bpd,
                                    'best_kl': best_kl,
                                    'best_nent': best_nent,
                                    'best_nepd': best_nepd},
                                   checkpoint_name)
                    try:
                        reconstruct(args, epoch, val_data, val_index, wolf)
                    except RuntimeError:
                        print('Reconstruction failed.')
                    try:
                        sample(args, epoch, wolf)
                    except RuntimeError:
                        print('Sampling failed')
            logging('Best NLL: {:.2f}, KL: {:.2f}, NENT: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}, epoch: {}'.format(
                best_nll, best_kl, best_nent, best_bpd, best_nepd, best_epoch), log)
            logging('=' * 125, log)
            # save checkpoint
            checkpoint_name = args.checkpoint_name + '{}.tar'.format(1)
            torch.save({'epoch': epoch + 1,
                        'step': -1,
                        'model': wolf.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_epoch': best_epoch,
                        'best_nll': best_nll,
                        'best_bpd': best_bpd,
                        'best_kl': best_kl,
                        'best_nent': best_nent,
                        'best_nepd': best_nepd},
                       checkpoint_name)


def main(args):
    args, (train_data, val_data), (train_index, val_index), wolf = setup(args)

    if is_master(args.rank):
        logging('# of Parameters: %d' % sum([param.numel() for param in wolf.parameters()]), args.log)
        if args.recover < 0:
            init_model(args, train_data, train_index, wolf)
            wolf.sync()

    if is_distributed(args.rank):
        wolf.init_distributed(args.rank, args.local_rank)

    train_loader, train_sampler, val_loader = init_dataloader(args, train_data, val_data)

    train(args, train_loader, train_index, train_sampler, val_loader, val_data, val_index, wolf)


if __name__ == "__main__":
    args = parse_args()
    assert args.rank == -1 and args.local_rank == 0, 'single process should have wrong rank ({}) or local rank ({})'.format(args.rank, args.local_rank)
    main(args)