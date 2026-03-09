"""
Adapted from https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py
Reference : Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test.
Adapted by Eliane Maalouf.
"""
from sys import stdout

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels


def MMD2u(K, m, n):
    """The MMD^2 unbiased statistic."""
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        + 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        - 2.0 / (m * n) * Kxy.sum()
    )


def MMD2b(K, m, n):
    """
    The MMD^2 biased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        1.0 / (m * m) * Kx.sum() + 1.0 / (n * n) * Ky.sum() - 2.0 / (m * n) * Kxy.sum()
    )


def MMD2(X, Y, kernel_params, unbiased=True):
    """
    Estimate the MMD^2 statistic between two samples.
    :param X: first sample
    :param Y: second sample
    :param kernel_params: dictionary with kernel parameters. Contains at least the following:
            - metric: string - kernel function to use. Default is 'rbf'
            other keys depend on the kernel function used. Example:
            - gamma: float - kernel bandwidth.
            For default 'rbf' kernel, gamma is the bandwidth. If not provided, it will be set using
            the median heuristic on the current sample.
    :param unbiased: boolean - whether to use the unbiased estimator of the MMD. Default is True.
    """
    m = len(X)
    n = len(Y)

    XY = np.vstack([X, Y])

    if kernel_params.get("metric", "rbf") == "rbf":
        if "gamma" not in kernel_params:
            print("kernel: RBF - estimating gamma using median heuristic.")
            sigma2 = estimate_median_pairwise_dists(
                XY, sample_ratio=1.0
            )  # does not count 0 distances in median
            # sigma2 = np.median(pairwise_distances(XY, metric='euclidean')) ** 2 # counts also 0 distances in median
            kernel_params["gamma"] = 1 / sigma2
            print(
                f"Estimated median pairwise squared distance: {sigma2}, setting gamma = {kernel_params['gamma']}"
            )
        else:
            print(f"kernel: RBF - using provided gamma = {kernel_params['gamma']}")

    K = pairwise_kernels(XY, **kernel_params)
    if unbiased:
        mmd_estimate = MMD2u(K, m, n)
    else:
        mmd_estimate = MMD2b(K, m, n)
    return mmd_estimate, K, kernel_params


def compute_null_distribution(
    K,
    m,
    n,
    iterations=10000,
    verbose=False,
    unbiased=True,
    random_state=None,
    marker_interval=1000,
):
    """Compute the bootstrap null-distribution of MMD2."""
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2_null[i] = MMD2u(K_i, m, n) if unbiased else MMD2b(K_i, m, n)

    if verbose:
        print("")

    return mmd2_null


def compute_null_distribution_given_permutations(
    K, m, n, permutation, unbiased=True, iterations=None
):
    """Compute the bootstrap null-distribution of MMD2u given
    predefined permutations.

    Note:: verbosity is removed to improve speed.
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        mmd2_null[i] = MMD2u(K_i, m, n) if unbiased else MMD2b(K_i, m, n)

    return mmd2_null


def kernel_two_sample_test(
    X,
    Y,
    kernel_function="rbf",
    iterations=10000,
    unbiased=True,
    verbose=False,
    random_state=None,
    **kwargs,
):
    """Compute MMD^2 (only relevant for unbiased), its null distribution and the p-value of the
    kernel two-sample test.

    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)

    obs_mmd, K, params = MMD2(X, Y, {"metric": kernel_function, **kwargs})
    if verbose:
        print(f"MMD^2 = {obs_mmd}")
        print("Computing the null distribution.")

    mmd2_null = compute_null_distribution(
        K,
        m,
        n,
        iterations,
        verbose=verbose,
        unbiased=unbiased,
        random_state=random_state,
    )
    p_value = max(1.0 / iterations, (mmd2_null > obs_mmd).sum() / float(iterations))
    if verbose:
        print(f"p-value ~= {p_value} \t (resolution : {1.0/iterations})")

    return obs_mmd, mmd2_null, p_value, params


def two_sample_mmd_test(
    X,
    Y,
    kernel_params,
    alpha=0.05,
    iterations=10000,
    unbiased=True,
    random_state=None,
    verbose=False,
):
    """Wrapper for kernel_two_sample_test() that takes a standardized kernel_params dictionary and returns the
    test statistic, p-value and H0 rejection decision.
    """
    if verbose:
        print(
            "\n Performing two sample test based on MMD statistic and permutation test. \n "
            "H0: The two samples come from the same distribution."
        )

        # raise warning if unbiased = False
        if not unbiased:
            print(
                "Warning: Using biased estimator for MMD. This may lead to incorrect results."
            )

        print(f"Alpha level: {alpha}")

    kernel_function = kernel_params.pop("metric", "rbf")
    obs_mmd, mmd2_null, p_value, params = kernel_two_sample_test(
        X,
        Y,
        kernel_function=kernel_function,
        iterations=iterations,
        unbiased=unbiased,
        verbose=verbose,
        random_state=random_state,
        **kernel_params,
    )
    if verbose:
        print(f"Kernel parameters: {params}")
    return obs_mmd, p_value, p_value < alpha


def estimate_median_pairwise_dists(all_data, sample_ratio=1.0, chunk_size=300):
    """
    Estimates the median of pairwise squared distances from the pooled data of two samples
    using a subsample to ensure memory efficiency.

    This is a chunked version that is logically equivalent to
    `estimate_median_pairwise_dists_2`: it uses all unique, non-diagonal
    pairwise distances between subsampled points (each unordered pair counted
    exactly once).

    Args:
        all_data: an array-like structure (e.g., NumPy array) containing all samples
                  for which pairwise distances are to be calculated.
        sample_ratio: Fraction of the full dataset to sample for distance calculation.
                      Default is 1.0 (use all data).
        chunk_size: Size of chunks to process at a time to avoid memory issues.

    Returns:
        float: The estimated median squared distance value.
    """
    all_data = np.asarray(all_data)

    # Flatten higher-dimensional inputs (e.g. images) to (n_samples, n_features)
    if all_data.ndim > 2:
        all_data = all_data.reshape(all_data.shape[0], -1)

    N_total = all_data.shape[0]

    if N_total < 2:
        # Not enough points to compute pairwise distances; return a sensible default
        return 1.0

    # Subsample indices
    N_subsample = min(int(N_total * sample_ratio), N_total)
    indices = np.random.choice(N_total, size=N_subsample, replace=False)
    D_subsample = all_data[indices, :]

    # Use chunked computation to avoid memory issues
    chunk_size = max(1, min(chunk_size, N_subsample))
    all_distances = []

    # Loop over row-chunks
    for i_start in range(0, N_subsample, chunk_size):
        i_end = min(i_start + chunk_size, N_subsample)
        block_i = D_subsample[i_start:i_end]

        # Loop over column-chunks (j_start >= i_start to avoid duplicate pairs)
        for j_start in range(i_start, N_subsample, chunk_size):
            j_end = min(j_start + chunk_size, N_subsample)
            block_j = D_subsample[j_start:j_end]

            # Compute block distances
            dists_block = pairwise_distances(
                block_i,
                block_j,
                metric="euclidean",
                squared=True,
            )

            if i_start == j_start:
                # Same block: use only the upper triangle (k=1) to avoid diagonal and duplicates
                m = dists_block.shape[0]
                if m > 1:
                    iu, ju = np.triu_indices(m, k=1)
                    all_distances.append(dists_block[iu, ju].ravel())
                # if m == 1, no pairs in this block
            else:
                # Different blocks: all distances correspond to unique (i, j) with i < j
                all_distances.append(dists_block.ravel())

    if not all_distances:
        return 1.0

    distances_sq = np.concatenate(all_distances)

    if distances_sq.size == 0:
        return 1.0

    return np.median(distances_sq)


def estimate_median_pairwise_dists_2(all_data, sample_ratio=0.1, max_pairs=500000):
    """
    Estimates the median of pairwise squared distances from the pooled data of two samples
    using a subsample to ensure memory efficiency.

    Args:
        all_data: an array-like structure (e.g., NumPy array) containing all samples for which
                   pairwise distances are to be calculated.
        sample_ratio (float): Fraction of the full dataset to sample for
                              distance calculation.
        max_pairs (int): Hard limit on the number of pairwise distances
                         to calculate (to ensure memory safety).

    Returns:
        float: The estimated median squared distance value.
    """

    # 1. Combine and Sample Data
    # Ensure all_data is a NumPy array
    all_data = np.asarray(all_data)

    # Flatten higher-dimensional inputs (e.g. images) to (n_samples, n_features)
    if all_data.ndim > 2:
        all_data = all_data.reshape(all_data.shape[0], -1)

    N_total = all_data.shape[0]

    # Determine the size of the subsample
    N_subsample = min(int(N_total * sample_ratio), N_total)

    # Get random indices for subsampling without replacement
    indices = np.random.choice(N_total, size=N_subsample, replace=False)
    D_subsample = all_data[indices, :]

    # 2. Calculate Distances on Subsample
    # This matrix size is N_subsample x N_subsample, much smaller than N_total x N_total

    # Use squared=True for squared Euclidean distance
    from sklearn.metrics.pairwise import euclidean_distances

    dist_matrix_sq = (
        pairwise_distances(D_subsample, D_subsample, metric="euclidean") ** 2
    )

    # 3. Flatten and Select
    # Get the upper triangle of the matrix to avoid zeros on the diagonal
    # and redundant symmetric distances.
    upper_tri_indices = np.triu_indices_from(dist_matrix_sq, k=1)
    distances_sq = dist_matrix_sq[upper_tri_indices]

    # To be extra safe, limit the number of distances to process (though unnecessary
    # if N_subsample is small enough).
    if distances_sq.size > max_pairs:
        distances_sq = np.random.choice(distances_sq, size=max_pairs, replace=False)

    # 4. Calculate Median Distance
    if distances_sq.size == 0:
        # Handle case where subsample size is too small
        return 1.0

    median_sq = np.median(distances_sq)  # Median of squared distances

    return median_sq


# test functions
def test_different_distributions(n=100, m=100, dim_x=10, dim_y=10, unbiased=True):
    # Generate samples from different distributions
    np.random.seed(0)
    X = np.random.randn(n, dim_x)  # Standard normal distribution
    Y = np.random.uniform(-2, 2, (m, dim_y))  # Uniform distribution

    alpha = 0.05
    kernel_function = "rbf"  # gamma will be set using median heuristic

    mmd2, mmd2_null, p_value, params = kernel_two_sample_test(
        X, Y, kernel_function=kernel_function, unbiased=unbiased, verbose=True
    )

    print("\nTest: Different Distributions (Normal vs. Uniform)")
    print(f"Test Statistic: {mmd2}")
    print(f"Params: {params}")
    if p_value < alpha:
        print("Result: Reject the null hypothesis (expected).")
    else:
        print("Result: Fail to reject the null hypothesis (unexpected).")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)

    m = 20
    n = 20
    d = 20

    unbiased = True  # use biased estimator or not

    sigma2X = np.eye(d)
    muX = np.zeros(d)

    sigma2Y = np.eye(d)
    # muY = np.ones(d)
    muY = np.zeros(d)

    iterations = 10000

    X = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=m)
    Y = np.random.multivariate_normal(mean=muY, cov=sigma2Y, size=n)

    if d == 2:
        plt.figure()
        plt.plot(X[:, 0], X[:, 1], "bo")
        plt.plot(Y[:, 0], Y[:, 1], "rx")

    # compare to estimate_median_pairwise_dists
    all_data = np.vstack((X, Y))
    sigma2 = (
        np.median(pairwise_distances(all_data, metric="euclidean")) ** 2
    )  # counts also 0 distances
    # test estimate_median_pairwise_dists
    sigma2_est = estimate_median_pairwise_dists(all_data, sample_ratio=1.0)
    sigma2_est2 = estimate_median_pairwise_dists_2(all_data, sample_ratio=1.0)
    print(f"Median pairwise squared distance (full): {sigma2}")
    print(f"Median pairwise squared distance (est): {sigma2_est}")
    print(f"Median pairwise squared distance (est 2): {sigma2_est2}")

    mmd2, mmd2_null, p_value, params = kernel_two_sample_test(
        X, Y, kernel_function="rbf", gamma=1.0 / sigma2, unbiased=unbiased, verbose=True
    )
    # mmd2u, mmd2u_null, p_value, params = kernel_two_sample_test(X, Y,
    #                                                   kernel_function='linear',
    #                                                   unbiased = unbiased,
    #                                                   verbose=True)

    plt.figure()
    prob, bins, patches = plt.hist(mmd2_null, bins=50, density=True)
    plt.plot(
        mmd2,
        prob.max() / 30,
        "w*",
        markersize=24,
        markeredgecolor="k",
        markeredgewidth=2,
        label=f"$MMD^2_u = {mmd2}$",
    )
    plt.xlabel("$MMD^2_u$")
    plt.ylabel("$p(MMD^2_u)$")
    plt.legend(numpoints=1)
    plt.title(f"$MMD^2_u$: null-distribution and observed value. $p$-value={p_value}")
    plt.show()

    test_different_distributions(n=50, m=50, dim_x=2000, dim_y=2000, unbiased=True)
