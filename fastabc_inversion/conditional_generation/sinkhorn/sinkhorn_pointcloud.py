#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

Source : https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py

"""

import torch
from torch.autograd import Variable


def sinkhorn_normalized(x, y, epsilon, n, niter, p=2, device=None):
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter, p, device)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter, p, device)
    Wxy = sinkhorn_loss(x, y, epsilon, n, niter, p, device)

    # print('Wxy:',Wxy,'Wxx:',Wxx, 'Wyy:', Wyy)
    return 2 * Wxy - Wxx - Wyy


def sinkhorn_loss(x, y, epsilon, n, niter, p=2, device="cpu"):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop.
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y, p)  # Wasserstein cost function

    # both marginals are fixed with equal weights
    mu = Variable(1.0 / n * torch.FloatTensor(n).fill_(1), requires_grad=False).to(
        device
    )
    nu = Variable(1.0 / n * torch.FloatTensor(n).fill_(1), requires_grad=False).to(
        device
    )

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -0.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10 ** (-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(
            torch.exp(A).sum(1, keepdim=True) + 1e-6
        )  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0.0 * mu, 0.0 * nu, 0.0
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.cpu().numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)

    return cost  # Sinkhorn cost


def cost_matrix(x, y, p=2):
    """Returns the matrix of $|x_i-y_j|^p$.
    Expects x and y in the shape (# of obs in the cloud, dimension of the data space).
    """

    x_col = x.unsqueeze(1)

    y_lin = y.unsqueeze(0)

    c = torch.sum(
        torch.abs(x_col - y_lin) ** p, 2
    )  # default form from original code (Peyré)
    # c = torch.sum(torch.abs(x_col - y_lin), 2) # form from Genevay in Sinkhorn_GAN
    return c
