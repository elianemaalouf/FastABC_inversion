"""
Adapted from https://github.com/aaron-xichen/pytorch-playground/blob/master/mnist/dataset.py
Adapted by Eliane Maalouf
"""

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import os


def get(
    batch_size,
    data_root="/tmp/public_dataset/pytorch",
    img_size=32,
    train=True,
    val=True,
    **kwargs
):
    data_root = os.path.expanduser(os.path.join(data_root, "mnist-data"))
    kwargs.pop("input_size", None)
    num_workers = kwargs.setdefault("num_workers", 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.5,), std=(0.5,)
                        )  # modified normalization by Eliane Maalouf
                        # transforms.Normalize((0.1307,), (0.3081,)) # original normalization
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
