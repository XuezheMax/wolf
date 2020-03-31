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
from PIL import Image
import signal
import threading
import multiprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image


from wolf.data import load_datasets, get_batch, preprocess, postprocess
from wolf import WolfModel
from experiments.options import parse_synthesize_args
from experiments.distributed import ErrorHandler


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
    num_class = 10 if dataset == 'cifar10' else None

    nc = 3
    args.nx = image_size ** 2 * nc
    n_bits = args.n_bits
    args.n_bins = 2. ** n_bits
    args.test_k = 5

    model_path = args.model_path
    result_path = os.path.join(model_path, 'synthesis')
    args.result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    data_path = args.data_path
    args.cuda = torch.cuda.is_available()

    random_seed = args.seed
    print('random seed: {}'.format(random_seed))
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if args.cuda:
            torch.cuda.manual_seed(random_seed)

    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True

    train_data, val_data = load_datasets(dataset, image_size, data_path=data_path)
    args.device = device
    wolf = WolfModel.load(model_path, device=device)
    wolf.eval()
    return args, wolf, (train_data, val_data, num_class)


def sample(args, wolf):
    print('sampling')
    wolf.eval()
    nsamples = args.nsamples
    n = 64 if args.image_size > 128 else 256
    tau = args.tau
    image_size = (3, args.image_size, args.image_size)

    nums = 0
    nnans = 0
    images = []

    make_grid = args.make_grid
    if make_grid:
        result_path = args.result_path
    else:
        result_path = os.path.join(args.result_path, str(tau))
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    start_time = time.time()
    while nums < nsamples:
        imgs = wolf.synthesize(n, image_size, tau=tau, n_bits=args.n_bits, device=args.device)
        mask = torch.isnan(imgs).view(n, -1).any(dim=1)
        nnans += mask.sum().item()
        imgs = imgs[torch.logical_not(mask)].cpu()
        if make_grid:
            images.append(imgs)
        else:
            for i in range(imgs.size(0)):
                img = imgs[i]
                img = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                image_file = 'sample{}.t{:.1f}.png'.format(i + nums, tau)
                im = Image.fromarray(img)
                im.save(os.path.join(result_path, image_file))

        nums += n - mask.sum().item()
        print(nums, nnans)

    if make_grid:
        imgs = torch.cat(images, dim=0)[:nsamples]
        nrow = int(math.sqrt(nsamples))
        image_file = 'sample.t{:.1f}.png'.format(tau)
        save_image(imgs, os.path.join(result_path, image_file), nrow=nrow)

    print('time: {:.1f}s'.format(time.time() - start_time))


def reconstruct(args, data, wolf):
    print('reconstruct')
    wolf.eval()
    batch = 16
    nsamples = 15

    index = np.arange(len(data))
    np.random.shuffle(index)
    img, y = get_batch(data, index[:batch])
    img = img.to(args.device)
    y = y.to(args.device)

    image_size = (3, args.image_size, args.image_size)
    _, epsilon = wolf.encode(img, y=y, n_bits=args.n_bits, nsamples=nsamples, random=True)
    epsilon = epsilon.view(batch * nsamples, *image_size)
    z = wolf.encode_global(img, y=y, n_bits=args.n_bits, nsamples=nsamples, random=True)
    z = z.view(batch * nsamples, z.size(2))
    # [batch, nsamples, c, h, w]
    img_recon = wolf.decode(epsilon, z=z, n_bits=args.n_bits).view(batch, nsamples, *image_size)
    # [batch, 1, c, h, w]
    img = postprocess(preprocess(img, args.n_bits), args.n_bits).unsqueeze(1)

    # [batch, nsamples + 1, c, h, w] -> [batch*(nsamples + 1), c, h, w]
    comparison = torch.cat([img, img_recon], dim=1).view(-1, *image_size).cpu()
    image_file = 'reconstruct.png'
    save_image(comparison, os.path.join(args.result_path, image_file), nrow=nsamples + 1)


def _interpolate(args, data, index, wolf, clabel):
    print('interpolate: {}, #{}'.format(clabel, len(index)))
    wolf.eval()
    batch = 64
    np.random.shuffle(index)

    img0, y0 = get_batch(data, index[:batch])
    img0 = img0.to(args.device)
    y0 = y0.to(args.device)
    img1, y1 = get_batch(data, index[batch:2 * batch])
    img1 = img1.to(args.device)
    y1 = y1.to(args.device)
    image_size = (3, args.image_size, args.image_size)

    z0, epsilon0 = wolf.encode(img0, y=y0, n_bits=args.n_bits, random=False)
    z1, epsilon1 = wolf.encode(img1, y=y1, n_bits=args.n_bits, random=False)

    alphas = [x * 0.1 for x in range(11)]
    # [1, time, 1, 1, 1]
    betas = torch.arange(11, device=args.device).float().view(1, 11, 1, 1, 1) * 0.1
    # [batch, time, dim]
    z0 = z0.expand(-1, betas.size(1), -1)
    z1 = z1.expand(-1, betas.size(1), -1)
    imgs = []
    for alpha in alphas:
        # [batch, time, dim]
        z = z0 * (1.0 - alpha) + z1 * alpha
        # [batch, time, c, h, w]
        epsilon = epsilon0 * (1.0 - betas) + epsilon1 * betas

        # [batch * time, *]
        z = z.view(-1, z.size(2))
        epsilon = epsilon.view(-1, *image_size)
        # [batch, time, c, h, w]
        img = wolf.decode(epsilon, z=z, n_bits=args.n_bits).view(batch, -1, *image_size)
        imgs.append(img)
    img = torch.stack(imgs, dim=1).view(-1, *image_size).cpu()
    image_file = 'interpolate{}.png'.format(clabel)
    save_image(img, os.path.join(args.result_path, image_file), nrow=11)


def interpolate(args, data, wolf, num_class):
    index = np.arange(len(data))
    _interpolate(args, data, index, wolf, '')
    if num_class is not None:
        index = np.arange(len(data))
        _, y = get_batch(data, index)
        index = torch.from_numpy(index)
        for label in range(num_class):
            mask = y.eq(label)
            idx = index[mask].numpy()
            _interpolate(args, data, idx, wolf, label)


def _switch(args, data, index, wolf, clabel):
    print('switch: {}, #{}'.format(clabel, len(index)))
    wolf.eval()
    batch = 64
    np.random.shuffle(index)

    for run in range(5):
        img0, y0 = get_batch(data, index[:batch])
        img0 = img0.to(args.device)
        y0 = y0.to(args.device)
        img1, y1 = get_batch(data, index[(run + 1) * batch:(run + 2) * batch])
        img1 = img1.to(args.device)
        y1 = y1.to(args.device)
        image_size = (3, args.image_size, args.image_size)

        z0, epsilon0 = wolf.encode(img0, y=y0, n_bits=args.n_bits, random=False)
        z1, epsilon1 = wolf.encode(img1, y=y1, n_bits=args.n_bits, random=False)

        alphas = torch.arange(2, device=args.device).float().view(1, 2, 1)
        # [1, time, 1, 1, 1]
        betas = [0, 1]
        # [batch, time, dim]
        epsilon0 = epsilon0.expand(-1, alphas.size(1), *image_size)
        epsilon1 = epsilon1.expand(-1, alphas.size(1), *image_size)
        imgs = []
        for beta in betas:
            # [batch, time, c, h, w]
            epsilon = epsilon0 * (1.0 - beta) + epsilon1 * beta
            # [batch, time, dim]
            z = z0 * (1.0 - alphas) + z1 * alphas if beta == 0 else z0 * alphas + z1 * (1.0 - alphas)

            # [batch * time, *]
            z = z.view(-1, z.size(2))
            epsilon = epsilon.view(-1, *image_size)
            # [batch, time, c, h, w]
            img = wolf.decode(epsilon, z=z, n_bits=args.n_bits).view(batch, -1, *image_size)
            imgs.append(img)

        nn = int(math.sqrt(batch))
        # [batch, 2, 2, c, h, w]
        img = torch.stack(imgs, dim=1)
        # [nn, nn, 2, 2, c, h, w] -> [nn, 2, nn, 2, c, h, w]
        img = img.view(nn, nn, 2, 2, *image_size).transpose(1, 2)
        img = img.contiguous().view(-1, *image_size).cpu()
        image_file = 'switch{}.png'.format(clabel + '-' + str(run))
        save_image(img, os.path.join(args.result_path, image_file), nrow=2 * nn)


def switch(args, data, wolf, num_class):
    index = np.arange(len(data))
    _switch(args, data, index, wolf, 'g')
    if num_class is not None:
        index = np.arange(len(data))
        _, y = get_batch(data, index)
        index = torch.from_numpy(index)
        for label in range(num_class):
            mask = y.eq(label)
            idx = index[mask].numpy()
            _switch(args, data, idx, wolf, label)


def classify(args, train_data, test_data, wolf):
    probe = args.probe
    wolf.eval()
    print('encoding')
    train_features, train_label = encode(args, train_data, wolf)
    test_features, test_label = encode(args, test_data, wolf)

    print('classifying')
    mp = multiprocessing.get_context('spawn')
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    processes = []
    for key in train_features:
        x_train = train_features[key]
        y_train = train_label
        x_test = test_features[key]
        y_test = test_label

        process = mp.Process(target=run_classifier,
                             args=(probe, key, x_train, y_train, x_test, y_test, error_queue),
                             daemon=False)
        process.start()
        error_handler.add_child(process.pid)
        processes.append(process)

    for process in processes:
        process.join()


def run_classifier(probe, key, x_train, y_train, x_test, y_test, error_queue):
    if probe == 'svm-rbf':
        clf = SVC(kernel='rbf')
    elif probe == 'svm-linear':
        clf = SVC(kernel='linear')
    elif probe == 'logistic':
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
    else:
        raise ValueError('unknown probe: {}'.format(probe))

    try:
        start = time.time()
        clf.fit(x_train.numpy(), y_train.numpy())
        acc = clf.score(x_test.numpy(), y_test.numpy()) * 100
        gc.collect()
        print("Dimensions on {}: {}, {}".format(key, tuple(x_train.size()), tuple(x_test.size())))
        print("Accuracy on {} is {:.2f}, time: {:.2f}s".format(key, acc, time.time() - start))
        print('-' * 25)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.rank, traceback.format_exc()))


def encode(args, data, wolf):
    batch_size = 64 if args.image_size > 128 else 256
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    device = args.device
    zs = []
    epsilons = []
    imgs = []
    labels = []
    for img, y in data_loader:
        imgs.append(img)
        labels.append(y)

        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z, epsilon = wolf.encode(img, y=y, n_bits=args.n_bits, random=False)
        if z is not None:
            zs.append(z.squeeze(1).cpu())
        epsilons.append(epsilon.squeeze(1).cpu())

    imgs = torch.cat(imgs, dim=0)
    imgs = imgs.view(imgs.size(0), -1)
    epsilons = torch.cat(epsilons, dim=0)
    epsilons = epsilons.view(epsilons.size(0), -1)
    labels = torch.cat(labels, dim=0)
    features = {'img': imgs, 'epsilon': epsilons}
    if len(zs) > 0:
        zs = torch.cat(zs, dim=0)
        features.update({'latent code': zs})
    return features, labels


def main(args):
    args, wolf, (train_data, val_data, num_class) = setup(args)
    if args.mode == 'sample':
        sample(args, wolf)
    elif args.mode == 'reconstruct':
        reconstruct(args, train_data, wolf)
    elif args.mode == 'interpolate':
        interpolate(args, train_data, wolf, num_class)
    elif args.mode == 'switch':
        switch(args, train_data, wolf, num_class)
    elif args.mode == 'classify':
        classify(args, train_data, val_data, wolf)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))


if __name__ == "__main__":
    args = parse_synthesize_args()
    with torch.no_grad():
        main(args)
