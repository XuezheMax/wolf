import os
import scipy.io
import numpy as np

import torch
from torchvision import datasets, transforms


def load_datasets(dataset, image_size, data_path):
    if dataset == 'omniglot':
        return load_omniglot()
    elif dataset == 'mnist':
        return load_mnist()
    elif dataset.startswith('lsun'):
        category = None if dataset == 'lsun' else dataset[5:]
        return load_lsun(data_path, category, image_size)
    elif dataset == 'cifar10':
        return load_cifar10(data_path)
    elif dataset == 'imagenet':
        return load_imagenet(data_path, image_size)
    elif dataset == 'celeba':
        return load_celeba(data_path, image_size)
    else:
        raise ValueError('unknown data set %s' % dataset)


def load_omniglot():
    def reshape_data(data):
        return data.T.reshape((-1, 1, 28, 28))

    omni_raw = scipy.io.loadmat('data/omniglot/chardata.mat')

    train_data = reshape_data(omni_raw['data']).astype(np.float32)
    train_label = omni_raw['target'].argmax(axis=0)
    test_data = reshape_data(omni_raw['testdata']).astype(np.float32)
    test_label = omni_raw['testtarget'].argmax(axis=0)

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_mnist():
    train_data, train_label = torch.load('data/mnist/processed/training.pt')
    test_data, test_label = torch.load('data/mnist/processed/test.pt')

    train_data = train_data.float().div(256).unsqueeze(1)
    test_data = test_data.float().div(256).unsqueeze(1)

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_lsun(data_path, category, image_size):
    if category is None:
        classes_train = 'train'
        classes_val = 'val'
    else:
        classes_train = [category + '_train']
        classes_val = [category + '_val']
    train_data = datasets.LSUN(data_path, classes=classes_train,
                               transform=transforms.Compose([
                                   transforms.CenterCrop(256),
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                               ]))

    val_data = datasets.LSUN(data_path, classes=classes_val,
                             transform=transforms.Compose([
                                 transforms.CenterCrop(256),
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                             ]))
    return train_data, val_data


def load_cifar10(data_path):
    imageSize = 32
    train_data = datasets.CIFAR10(data_path, train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4, padding_mode='reflect'),
                                      transforms.RandomCrop(imageSize),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor()
                                  ]))
    test_data = datasets.CIFAR10(data_path, train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    return train_data, test_data


def load_imagenet(data_path, image_size):
    data_path = os.path.join(data_path, 'imagenet{}x{}'.format(image_size, image_size))
    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
    val_data = datasets.ImageFolder(os.path.join(data_path, 'val'),
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
    return train_data, val_data


def load_celeba(data_path, image_size):
    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                      transform=transforms.Compose([
                                          transforms.Resize(image_size),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor()
                                      ]))
    val_data = datasets.ImageFolder(os.path.join(data_path, 'val'),
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor()
                                    ]))
    return train_data, val_data


def get_batch(data, indices):
    imgs = []
    labels = []
    for index in indices:
        img, label = data[index]
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs, dim=0), torch.LongTensor(labels)


def iterate_minibatches(data, indices, batch_size, shuffle):
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield get_batch(data, excerpt)


def binarize_image(img):
    return torch.rand(img.size()).type_as(img).le(img).float()


def binarize_data(data):
    return [(binarize_image(img), label) for img, label in data]


def preprocess(img, n_bits, noise=None):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    if noise is not None:
        # [batch, nsamples, channels, H, W]
        img = img.unsqueeze(1) + noise
    # normalize
    img = img.div(n_bins)
    img = (img - 0.5).div(0.5)
    return img


def postprocess(img, n_bits):
    n_bins = 2. ** n_bits
    # re-normalize
    img = img.mul(0.5) + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img
