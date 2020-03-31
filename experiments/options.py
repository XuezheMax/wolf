import os, sys
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Wolf')
    parser.add_argument('--rank', type=int, default=-1, metavar='N',
                        help='rank of the process in all distributed processes')
    parser.add_argument("--local_rank", type=int, default=0, metavar='N',
                        help='rank of the process in the machine')
    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--eval_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for eval (default: 4)')
    parser.add_argument('--batch_steps', type=int, default=1, metavar='N',
                        help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
    parser.add_argument('--init_batch_size', type=int, default=1024, metavar='N',
                        help='number of instances for model initialization (default: 1024)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--valid_epochs', type=int, default=1, metavar='N',
                        help='number of epochs to validate model (default: 1)')
    parser.add_argument('--seed', type=int, default=65537, metavar='S',
                        help='random seed (default: 65537)')
    parser.add_argument('--train_k', type=int, default=1, metavar='N',
                        help='training K (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, metavar='N',
                        help='number of steps to warm up (default: 500)')
    parser.add_argument('--lr_decay', type=float, default=0.999997, help='Decay rate of learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--grad_clip', type=float, default=0,
                        help='max norm for gradient clip (default 0: no clip')
    parser.add_argument('--dataset', choices=['cifar10', 'lsun', 'imagenet', 'celeba'],
                        help='data set', required=True)
    parser.add_argument('--category', choices=[None, 'bedroom', 'tower', 'church_outdoor'],
                        help='category', default=None)
    parser.add_argument('--image_size', type=int, required=True, metavar='N',
                        help='input image size')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--n_bits', type=int, default=8, metavar='N',
                        help='number of bits per pixel.')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--recover', type=int, default=-1, help='recover the model from disk.')

    return parser.parse_args()


def parse_distributed_args():
    parser = ArgumentParser(description="Dist Wolf")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--eval_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for eval (default: 4)')
    parser.add_argument('--batch_steps', type=int, default=1, metavar='N',
                        help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
    parser.add_argument('--init_batch_size', type=int, default=1024, metavar='N',
                        help='number of instances for model initialization (default: 1024)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--valid_epochs', type=int, default=1, metavar='N',
                        help='number of epochs to validate model (default: 1)')
    parser.add_argument('--seed', type=int, default=65537, metavar='S',
                        help='random seed (default: 65537)')
    parser.add_argument('--train_k', type=int, default=1, metavar='N',
                        help='training K (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, metavar='N',
                        help='number of steps to warm up (default: 500)')
    parser.add_argument('--lr_decay', type=float, default=0.999997, help='Decay rate of learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--grad_clip', type=float, default=0,
                        help='max norm for gradient clip (default 0: no clip')
    parser.add_argument('--dataset', choices=['cifar10', 'lsun', 'imagenet', 'celeba'],
                        help='data set', required=True)
    parser.add_argument('--category', choices=[None, 'bedroom', 'tower', 'church_outdoor'],
                        help='category', default=None)
    parser.add_argument('--image_size', type=int, required=True, metavar='N',
                        help='input image size')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--n_bits', type=int, default=8, metavar='N',
                        help='number of bits per pixel.')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--recover', type=int, default=-1, help='recover the model from disk.')

    return parser.parse_args()


def parse_synthesize_args():
    parser = ArgumentParser(description="Wolf Synthesize")
    parser.add_argument('--mode', choices=['sample', 'reconstruct', 'interpolate', 'switch', 'classify'], help='synthesis mode', required=True)
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed')
    parser.add_argument('--dataset', choices=['cifar10', 'lsun', 'imagenet', 'celeba'], help='data set', required=True)
    parser.add_argument('--category', choices=[None, 'bedroom', 'tower', 'church_outdoor'],
                        help='category', default=None)
    parser.add_argument('--image_size', type=int, required=True, metavar='N',
                        help='input image size')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--n_bits', type=int, default=8, metavar='N',
                        help='number of bits per pixel.')
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--tau', type=float, default=1.0, metavar='S', help='temperature for iw decoding (default: 1.0)')
    parser.add_argument('--nsamples', type=int, default=256, metavar='N', help='number of samples.')
    parser.add_argument('--make_grid', action='store_true', help='make grid of image')
    parser.add_argument('--probe', choices=['svm-linear', 'svm-rbf', 'logistic'], default=None, help='classifier for probe')

    return parser.parse_args()
