import numpy as np
import torch
from scipy.spatial import distance


def dss(target, mean, cov, upper_cond=1e10, ignore_cond=False, verbose=False):
    """
    Computes Dawid-Sebastiani score to evaluate the distance between a target vector and a Gaussian distribution
    given by its mean and covariance.
    Note: as a comparison between distributions, this score represents the distance between a Dirac mass on
     the target and a Gaussian(mean, cov)

    :param target: a p-dim array of shape (1,p) containing the target vector
    :param mean: a p-dim array of shape (1,p) containing the mean of the Gaussian distribution
    :param cov: a p x p array containing the covariance matrix of the Gaussian distribution
    :param upper_cond: a scalar value representing the upper bound for the condition number of the covariance matrix
    :param ignore_cond: a boolean value indicating whether to ignore the condition number of the covariance matrix
    :param verbose: a boolean value indicating whether to print some information about the computation
    :return: a scalar value representing the distance between the target and the distribution
    """

    # verify that the target and the mean have the same dimension
    assert target.shape == mean.shape

    # invert cov
    try:
        L = np.linalg.cholesky(cov)

        # Compute the inverse using Cholesky decomposition
        L_inv = np.linalg.inv(L)
        cov_inv = np.dot(L_inv.T, L_inv)
    except np.linalg.LinAlgError:
        print(
            "DSS computation: The covariance matrix is not positive definite. The score is set to None."
        )
        return None

    # svd_cov = sLA.svd(cov, full_matrices=True)
    # singular_values = svd_cov[1]
    # eigen_vectors = svd_cov[0]

    # compute the determinant of the covariance matrix
    det = np.linalg.det(cov)

    # replace zero determinant by the smallest value 1e-323 that makes the log different from -inf
    # but do this only if cov has a condition number below the upper bound and full rank
    if det == 0.0 and np.linalg.matrix_rank(cov) == cov.shape[0]:
        if ignore_cond or np.linalg.cond(cov) < upper_cond:
            det = 1e-323
            if verbose:
                print(
                    f"DSS computation: The covariance matrix has zero determinant, but still has full rank."
                    f"{'Condition number check was ignored ' if ignore_cond else 'Matrix condition number below the upper bound:'}"
                    f"{str(upper_cond) if not ignore_cond else ''}."
                    f"The zero determinant is replaced by 1e-323 to avoid -inf in the log."
                )
    elif np.linalg.matrix_rank(cov) != cov.shape[0] or (
        not ignore_cond and np.linalg.cond(cov) > upper_cond
    ):
        print(
            "DSS computation: The covariance matrix is either extremely ill-conditioned, "
            "or not full rank. The score is set to None."
        )
        return None
    else:
        if det == np.inf:
            det = np.finfo(np.float64).max
            if verbose:
                print(
                    f"DSS computation: The covariance matrix has determinant +inf, but still has full rank "
                    f"and a condition number below the upper bound. The determinant is replaced by the maximum float64 value "
                    f"{str(np.finfo(np.float64).max)}"
                )
        else:
            if verbose:
                print(f"DSS computation: The determinant of covariance is {det}.")

    # compute the inverse of the covariance matrix
    # cov_inv = np.matmul(eigen_vectors, np.matmul(np.diag(1/singular_values), eigen_vectors.T))

    # verify that the inverse is correct
    np.allclose(np.eye(cov.shape[0]), np.matmul(cov, cov_inv))

    # compute the distance between the target and the distribution
    dist = distance.mahalanobis(target, mean, cov_inv) ** 2

    # return the score
    return np.log(det) + dist, det, dist


def dss_fast(
    target, mean, cov_inv, cov_det, cov_cond, cov_rank, upper_cond=1e10, verbose=False
):
    """
    Computes Dawid-Sebastiani score to evaluate the distance between a target vector and a Gaussian distribution
    given by its mean and covariance.
    Note: as a comparison between distributions, this score represents the distance between a Dirac mass on
     the target and a Gaussian(mean, cov)

    :param target: a p-dim array of shape (1,p) containing the target vector
    :param mean: a p-dim array of shape (1,p) containing the mean of the Gaussian distribution
    :param cov: a p x p array containing the covariance matrix of the Gaussian distribution
    :param upper_cond: a scalar value representing the upper bound for the condition number of the covariance matrix
    :return: a scalar value representing the distance between the target and the distribution
    """

    # verify that the target and the mean have the same dimension
    assert target.shape == mean.shape

    # replace zero determinant by the smallest value 1e-323 that makes the log different from -inf
    # but do this only if cov has a condition number below the upper bound and full rank
    if cov_det == 0.0 and cov_cond < upper_cond and cov_rank == cov_inv.shape[0]:
        cov_det = 1e-323
        if verbose:
            print(
                f"DSS computation: The covariance matrix has zero determinant, but still has full rank "
                f"and a condition number below the upper bound {str(upper_cond)}. The zero determinant is replaced by 1e-323 "
                f"to avoid -inf in the log."
            )
    elif cov_cond > upper_cond or cov_rank != cov_inv.shape[0]:
        print(
            "DSS computation: The covariance matrix is either extremely ill-conditioned, "
            "or not full rank. The score is set to None."
        )
        return None
    else:
        if cov_det == np.inf:
            cov_det = np.finfo(np.float64).max
            if verbose:
                print(
                    f"DSS computation: The covariance matrix has determinant +inf, but still has full rank "
                    f"and a condition number below the upper bound. The determinant is replaced by the maximum float64 value "
                    f"{str(np.finfo(np.float64).max)}"
                )
        else:
            if verbose:
                print(f"DSS computation: The determinant of covariance is {cov_det}.")

    # compute the distance between the target and the distribution
    dist = (distance.mahalanobis(target, mean, cov_inv)) ** 2

    # return the score
    return np.log(cov_det) + dist, cov_det, dist


def es(observation, samples, power=2, use_ks=True):
    """
    Implements empirical estimation of the Energy score.
    :param observation: observation vector of shape (1,dim)
    :param samples: sample from the predictive distribution to test of shape (n_samples,dim)
    :param power: power for the distance
    :return: energy score
    """

    # verify that the observation and the samples have the same dimension
    assert observation.shape[1] == samples.shape[1]

    n_samples = samples.shape[0]
    dim = samples.shape[1]

    if not use_ks:
        # compute the p-norm between the observation and the samples using the power
        dist = np.mean(np.sum(np.abs((samples - observation)) ** power, axis=1))

        # computes pairwise p-norm between samples using the power
        dist_pairwise = (
            np.abs(
                (
                    samples.reshape((n_samples, 1, dim))
                    - samples.reshape((1, n_samples, dim))
                )
            )
            ** power
        )
        dist_pairwise = np.sum(dist_pairwise, axis=2)
        dist_pairwise = np.sum(dist_pairwise.reshape(-1)) / (2 * n_samples**2)

        # compute the score
        return dist - dist_pairwise
    else:
        if power == 2:
            kernel_es = pairwise_distances
            kwargs_es = {"metric": "euclidean", "squared": True}
        if power == 1:
            kernel_es = pairwise_distances
            kwargs_es = {"metric": "l1"}  # {"power": 1}

        return ks(observation, samples, kernel=kernel_es, **kwargs_es)


def torch_es(observation, samples, power=2, on_gpu=True):
    """
    Estimates the Energy Score using PyTorch tensors for potential GPU acceleration.
    Args:
        observation (torch.Tensor): The observed outcome tensor.
                                    Shape: (1, dim)
        samples (torch.Tensor): The tensor of forecast samples.
                                Shape: (n_samples, dim)
        power (int, optional): The exponent for the p-norm. Defaults to 2 (Euclidean norm).
        on_gpu (bool, optional): If True, moves tensors to GPU if available. Defaults to True.

    Returns:
        torch.Tensor: The calculated energy score as a scalar tensor.
                      Call .item() on the result to get a Python number.
    """
    # verify that the observaiton and the samples are torch tensors and transform them if necessary
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)

    if on_gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # Move tensors to the specified device if not already there
        observation = observation.to(device)
        samples = samples.to(device)

    # Verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension"

    n_samples = samples.shape[0]
    dim = samples.shape[1]

    # Compute the mean p-norm between the observation and the samples
    # The use of .abs() is equivalent to np.abs()
    # The use of .sum(dim=1) is equivalent to np.sum(axis=1)
    dist = torch.mean(torch.sum(torch.abs(samples - observation) ** power, dim=1))

    # Compute the mean pairwise p-norm between all samples
    # The broadcasting trick with reshape works identically in PyTorch
    pairwise_diff = (
        torch.abs(samples.view(n_samples, 1, dim) - samples.view(1, n_samples, dim))
        ** power
    )

    # Sum across the feature dimension (dim=2)
    dist_pairwise = torch.sum(pairwise_diff, dim=2)

    # Sum all pairwise distances and normalize.
    # The division by 2 corrects for double-counting (dist(A,B) = dist(B,A)).
    dist_pairwise = torch.sum(dist_pairwise) / (2 * n_samples**2)

    # Compute the final score
    return (dist - dist_pairwise).item()


def vs(observation, samples, power=0.5, w=None):
    """
    Compute a score based on the pairwise differences between components of
    a single observation and multiple samples, weighted by a matrix.

    The function takes a single observation and a set of samples, calculates
    pairwise differences for their components raised to the specified power,
    and computes a weighted score based on these differences. A weight matrix
    is used for the computation, ensuring only non-negative values for weights.

    :param observation: A single observation with shape (1, d), where `d` is
        the dimensionality of the data.
    :param samples: A collection of samples with shape (n, d), where `n` is the
        number of samples and `d` is the dimensionality of the data.
    :param power: The power to which pairwise differences are raised. Defaults to 0.5.
    :param w: An optional weight matrix of shape (d, d). Defaults to a matrix with
        all elements set to 1. Must be non-negative if provided.
    :return: A scalar score computed based on the pairwise differences and weight matrix.
    """

    # verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    n_samples = samples.shape[0]
    dim = samples.shape[1]

    # if w is None, fill it with ones
    if w is None:
        w = np.ones((dim, dim))
    else:
        # verify that none of the elements in w is negative and that w has the correct shape
        assert np.all(w >= 0) and w.shape == (dim, dim)

    # compute the pairwise differences between the observation components
    diff_obs = (
        np.abs((observation.reshape((1, dim)) - observation.reshape((dim, 1)))) ** power
    )

    # compute the pairwise differences between the samples' components
    diff_samples = np.mean(
        np.abs(
            (
                samples.reshape((n_samples, dim, 1))
                - samples.reshape((n_samples, 1, dim))
            )
        )
        ** power,
        axis=0,
    )

    # compute the score
    return np.sum(w * (diff_obs - diff_samples) ** 2)


def torch_vs(observation, samples, power=0.5, w=None, on_gpu=True):
    """
    Estimates the Variogram Score using PyTorch tensors for potential GPU acceleration.

    This score evaluates a forecast by comparing the internal component-wise
    differences of the observation with the average internal component-wise
    differences of the forecast samples.

    Args:
        observation (torch.Tensor): The observed outcome tensor.
                                    Shape: (1, dim)
        samples (torch.Tensor): The tensor of forecast samples.
                                Shape: (n_samples, dim)
        power (float, optional): The exponent for the variogram. Defaults to 0.5.
        w (torch.Tensor, optional): A weight matrix to emphasize differences
                                    between certain components. Must be on the
                                    same device as the other tensors.
                                    Shape: (dim, dim). Defaults to ones.
        on_gpu (bool, optional): If True, moves tensors to GPU if available.

    Returns:
        torch.Tensor: The calculated variogram score as a scalar tensor.
                      Call .item() on the result to get a Python number.
    """
    # verify that the observaiton and the samples are torch tensors and transform them if necessary
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)

    if on_gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # Move tensors to the specified device if not already there
        observation = observation.to(device)
        samples = samples.to(device)

    # Verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    n_samples = samples.shape[0]
    dim = samples.shape[1]

    # If w is None, fill it with ones on the same device as the samples
    if w is None:
        w = torch.ones((dim, dim), device=samples.device)
    else:
        # Verify that w has the correct shape and no negative elements
        assert w.shape == (dim, dim) and torch.all(w >= 0)

    # Term 1: Compute pairwise differences between the observation's components
    # The .view() method is a PyTorch equivalent to reshape()
    diff_obs = (torch.abs(observation.view(1, dim) - observation.view(dim, 1))) ** power

    # Term 2: Compute mean of pairwise differences between the samples' components
    # The broadcasting trick works identically. Use dim=0 for mean over samples.
    diff_samples = torch.mean(
        torch.abs(samples.view(n_samples, dim, 1) - samples.view(n_samples, 1, dim))
        ** power,
        dim=0,
    )

    # Compute the final weighted score
    return torch.sum(w * (diff_obs - diff_samples) ** 2).item()


def ks(observation, samples, kernel=None, **kwargs):
    """
    Computes the Kernel Score (KS) as a measure of similarity between an observation and a set
    of samples, using a specified kernel function. The function calculates pairwise kernel values
    and determines the score based on the mean values of these computations.

    :param observation: A single data point of shape (1, d), where d is the dimension of the data.
    :param samples: A set of samples of shape (n, d), where n is the number of samples and d is
                    the dimension of the data.
    :param kernel: Callable or None, a kernel function to compute similarities. If None, the
                   function attempts to use the squared Euclidean distance from sklearn.
    :param kwargs: Additional keyword arguments to pass to the kernel function. Specific arguments
                   such as 'metric' and 'squared' may be included when using the default kernel.

    :return: The kernel score, computed as the difference between the mean kernel similarity of
             the observation and samples, and the pairwise kernel similarity within the samples.
    :rtype: float
    """
    # verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    if kernel is None:
        # try to import the pairwise distance function from sklearn
        try:
            from sklearn.metrics import pairwise_distances

            kernel = pairwise_distances
            # add to **kwargs the metric to use and the squared flag
            kwargs["metric"] = "euclidean"
            kwargs["squared"] = True  # squared Euclidean distance
            print("Using the Euclidean distance as a kernel.")
        except:
            raise ImportError(
                "The kernel function is None and the import of the Euclidean function from sklearn failed. "
                "Please provide a kernel function."
            )

    n_samples = samples.shape[0]

    # compute the kernel between the observation and the samples
    k_obs = kernel(observation, samples, **kwargs).reshape(-1)
    k_obs = np.mean(k_obs)

    # computes pairwise kernel between samples
    K_pairwise = kernel(samples, samples, **kwargs).reshape(-1)
    K_pairwise = np.sum(K_pairwise) / (2 * n_samples**2)

    # compute the score
    return k_obs - K_pairwise


def rbf_maha(observation, sample, cov=None):
    """
    Computes the Mahalanobis distance between an observation and a sample using the RBF kernel.
    :param observation: observation vector of shape (1,dim)
    :param sample: sample from the predictive distribution to test of shape (n_samples,dim)
    :param cov: covariance matrix of shape (dim,dim)
    :return: Gaussian kernel score
    """

    # invert cov
    try:
        L = np.linalg.cholesky(cov)

        # Compute the inverse using Cholesky decomposition
        L_inv = np.linalg.inv(L)
        cov_inv = np.dot(L_inv.T, L_inv)
    except np.linalg.LinAlgError as err:
        print(err.message)
        return None

    # verify that the inverse is correct
    np.allclose(np.eye(cov.shape[0]), np.matmul(cov, cov_inv))

    # compute the kernel between the observation and the samples
    maha_obs = []
    for i in range(sample.shape[0]):
        maha_obs.append(distance.mahalanobis(observation, sample[i, :], cov_inv) ** 2)
    maha_obs = np.array(maha_obs)
    maha_obs_mean = np.mean(maha_obs)
    maha_obs = np.exp(-maha_obs)
    maha_obs = np.mean(maha_obs)

    # computes pairwise kernel between samples
    maha_pairwise = []
    for i in range(sample.shape[0]):
        for j in range(sample.shape[0]):
            maha_pairwise.append(
                distance.mahalanobis(sample[i, :], sample[j, :], cov_inv) ** 2
            )
    maha_pairwise = np.array(maha_pairwise)
    maha_pairwise_mean = np.sum(maha_pairwise) / (2 * sample.shape[0] ** 2)
    maha_pairwise = np.exp(-maha_pairwise)
    maha_pairwise = np.sum(maha_pairwise) / (2 * sample.shape[0] ** 2)

    rbf_score = maha_obs - maha_pairwise
    maha_score = maha_obs_mean - maha_pairwise_mean

    # compute the score
    return rbf_score, maha_score


def rmse(observation, samples):
    """
    Calculate the Root Mean Square Error (RMSE) between an observation and multiple samples.

    The RMSE is a standard way to measure the difference between a single
    observation and a set of predicted data points (samples). This function
    expects the observation to be a single sample and computes the RMSE for
    each sample in the given set of samples.

    :param observation: A single sample with dimensions (1, D) representing
        the true values.
    :param samples: A 2D array with dimensions (N, D), where N is the number
        of samples and D is the dimensionality of each sample.

    :return: A 1D array of shape (N,) containing the RMSE values for each
        sample in the samples array compared to the observation.
    """
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    return np.sqrt(np.mean((observation - samples) ** 2, axis=1))


def torch_rmse(observation, samples, on_gpu=True):
    """
    Calculates the Root Mean Square Error (RMSE) between an observation and multiple samples.

    Args:
        observation (torch.Tensor): A 2D tensor representing a single observation.
                                    Expected shape: (1, features).
        samples (torch.Tensor): A 2D tensor representing multiple samples.
                                Expected shape: (num_samples, features).
        on_gpu (bool, optional): If True, moves tensors to GPU if available. Defaults to True.

    Returns:
        torch.Tensor: A 1D tensor containing the RMSE calculated along the feature dimension.
                      The shape will be (num_samples,).
    """
    # verify that the observaiton and the samples are torch tensors and transform them if necessary
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)

    if on_gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # Move tensors to the specified device if not already there
        observation = observation.to(device)
        samples = samples.to(device)

    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same number of features (dimension 1)."
    assert (
        observation.shape[0] == 1
    ), "Observation must be a single sample (dimension 0 must be 1)."

    rmse_results = torch.sqrt(torch.mean((observation - samples) ** 2, dim=1))

    if on_gpu:
        rmse_results = rmse_results.cpu().numpy()
    else:
        rmse_results = rmse_results.numpy()

    return rmse_results


# test rbf_maha function and measure time
if __name__ == "__main__":
    """
    import time as time

    # 3D covariance matrix
    cov = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    # 3D observation
    obs = np.array([0, 0, 0])
    # 3D sample from multi-variate Gaussian with covariance cov
    sample = np.random.multivariate_normal(np.zeros(3), cov, 100)

    # compute the score and measure time
    start = time.time()
    rbf_score, maha_score = rbf_maha(obs, sample, cov)
    end = time.time()
    print(f"Score: {rbf_score}, {maha_score}, time: {end - start}")

    # test energy score function name
    print(es.__name__)
    """

    # compare ks() to es() to torch_es()
    obs = np.array([[10, 2, 9]])
    samples = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 5]])

    from sklearn.metrics import pairwise_distances

    print("Energy Score (ES):")
    print(ks(obs, samples, kernel=pairwise_distances, metric="l1"))
    print(es(obs, samples, power=1, use_ks=True))
    print(torch_es(obs, samples, power=1, on_gpu=True))

    print("Variogram Score (VS):")
    print(vs(obs, samples, power=0.5))
    print(torch_vs(obs, samples, power=0.5, on_gpu=True))

    print("RMSE:")
    print(rmse(obs, samples))
    print(torch_rmse(obs, samples, on_gpu=True))
