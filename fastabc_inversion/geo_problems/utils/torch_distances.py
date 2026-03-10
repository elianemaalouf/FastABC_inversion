"""
Some commonly used distance/error functions
"""

import torch


def lpp_torch(x, y, p, keepdim=False):
    """
    Compute the lp, to the power p, distance between two tensors of same shape.
    Tensors must be on the same device and are expected
    to be of shape (n, d) where n is the number of samples and d is the dimensionality of the samples.
    :param x: tensor 1
    :param y: tensor 2
    :param p: power value for the lp distance
    :param keepdim: give the output the same shape as x and y
    :return: lp distance between x and y
    """
    # check that both x and y are tensors, and make them tensors if not
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    # check that both tensors are on the same device
    if x.device != y.device:
        # put y on the same device as x
        y = y.to(x.device)

    return torch.sum(torch.abs(x - y) ** p, dim=1, keepdim=keepdim)


def lp_torch(x, y, p=2, keepdim=False):
    """
    Compute the lp distance between two tensors of same shape. Tensors must be on the same device and are expected
    to be of shape (n, d) where n is the number of samples and d is the dimensionality of the samples.
    :param x: tensor 1
    :param y: tensor 2
    :param p: power value for the lp distance
    :param keepdim: give the output the same shape as x and y
    :return: lp distance between x and y
    """
    return lpp_torch(x=x, y=y, p=p, keepdim=keepdim) ** (1 / p)


def mse_torch(x, y, p=2, keepdim=False):
    """
    Mean squared error between two tensors of same shape. Tensors must be on the same device and are expected
    to be of shape (n, d) where n is the number of samples and d is the dimensionality of the samples.
    :param x: tensor 1
    :param y: tensor 2
    :param keepdim: give the output the same shape as x and y
    :return: mean squared error between x and y
    """
    # check that both x and y are tensors, and make them tensors if not
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    # check that both tensors are on the same device
    if x.device != y.device:
        # put y on the same device as x
        y = y.to(x.device)

    dim = x.shape[1]
    return lpp_torch(x=x, y=y, p=p, keepdim=keepdim) / dim
    # return torch.mean((x - y) ** 2, dim=1, keepdim=keepdim)


def rmse_torch(x, y, keepdim=False):
    """
    Root mean squared error between two tensors of same shape. Tensors must be on the same device and are expected
    to be of shape (n, d) where n is the number of samples and d is the dimensionality of the samples.
    :param x: tensor 1
    :param y: tensor 2
    :param keepdim: give the output the same shape as x and y
    :return: root mean squared error between x and y
    """
    return torch.sqrt(mse_torch(x, y, p=2, keepdim=keepdim))


# test
if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
    y = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.float32)

    print(lpp_torch(x, y, p=1, keepdim=False))
    print(lp_torch(x, y, p=1))
    print(mse_torch(x, y, p=1))
    print(rmse_torch(x, y))
