"""
COVARIANCE MATRICES
-------------------

Module containing methods to fill in covariance matrices given horizontal and vertical differences between
coordinates on a grid

"""
import numpy as np


def create_covMatrix(xx, yy, lxtrue, lytrue, sigma2, type="exponential", jitter=0):
    """create_covMatrix

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    if type == "exponential":
        return exponential(xx, yy, lxtrue, lytrue, sigma2, jitter)
    elif type == "matern32":
        return matern32(xx, yy, lxtrue, lytrue, sigma2, jitter)
    elif type == "matern52":
        return matern52(xx, yy, lxtrue, lytrue, sigma2, jitter)
    elif type == "gaussian":
        return gaussian(xx, yy, lxtrue, lytrue, sigma2, jitter)
    else:
        return None


def exponential(xx, yy, lxtrue, lytrue, sigma2, jitter):
    """exponential
    Exponential covariance kernel

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    ntot = xx.shape[0]
    Hm = np.sqrt(np.power(xx / lxtrue, 2) + np.power(yy / lytrue, 2))
    CM = np.exp(-np.abs(Hm)) * sigma2 + np.identity(ntot) * jitter  # covariance matrix
    return CM


def matern32(xx, yy, lxtrue, lytrue, sigma2, jitter):
    """matern32
    Matern 3/2 covariance kernel

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    ntot = xx.shape[0]
    Hm = np.sqrt(np.power(xx / lxtrue, 2) + np.power(yy / lytrue, 2))
    CM = (
        sigma2 * (1 + np.sqrt(3) * Hm) * np.exp(-np.sqrt(3) * Hm)
        + np.identity(ntot) * jitter
    )  # covariance matrix
    return CM


def matern52(xx, yy, lxtrue, lytrue, sigma2, jitter):
    """matern52
    Matern 5/2 covariance kernel

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    ntot = xx.shape[0]
    Hm = np.sqrt(np.power(xx / lxtrue, 2) + np.power(yy / lytrue, 2))
    CM = (
        sigma2 * (1 + np.sqrt(5) * Hm + (5 / 3) * Hm**2) * np.exp(-np.sqrt(5) * Hm)
        + np.identity(ntot) * jitter
    )  # covariance matrix
    return CM


def gaussian(xx, yy, lxtrue, lytrue, sigma2, jitter):
    """gaussian
    Gaussian covariance kernel

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    ntot = xx.shape[0]
    Hm = 0.5 * (np.power(xx / lxtrue, 2) + np.power(yy / lytrue, 2))
    CM = np.exp(-np.abs(Hm)) * sigma2 + np.identity(ntot) * jitter  # covariance matrix
    return CM


def prior_setup(nx, ny, lxtrue, lytrue, sigma2, cov_kernel, jitter, mu):
    ## grid
    ntot = nx * ny
    xx, yy = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    # print('xx: ', xx.shape)
    # print('yy:', yy.shape)

    xf = xx.flatten(
        order="C"
    )  # flatten 2D to 1D, reading from left to right, columns first then rows
    yf = yy.flatten(order="C")

    ## coordinates differences
    xx = np.subtract.outer(
        xf, xf
    )  # nx by nx matrix containing all 2 by 2 differences between horizontal coordinates
    yy = np.subtract.outer(
        yf, yf
    )  # ny by ny matrix containing all 2 by 2 differences between vertical coordinates

    ## create covariance matrix:
    CM = create_covMatrix(
        xx, yy, lxtrue, lytrue, sigma2, type=cov_kernel, jitter=jitter
    )  # Gaussian field prior covariance matrix
    L = np.linalg.cholesky(CM)  # square root of prior covariance matrix

    ## mean vector
    m_prior = np.repeat(mu, ntot)  # Gaussian filed prior mean

    return {
        "ntot": ntot,
        "covariance_matrix": CM,
        "covM_squareRoot": L,
        "prior_mean": m_prior,
    }
