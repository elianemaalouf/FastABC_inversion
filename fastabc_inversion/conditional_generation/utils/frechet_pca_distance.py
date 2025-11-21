import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


def calculate_frechet_distance(act1, act2):
    """
    Calculates the Fréchet distance between two multivariate Gaussians
    represented by their feature activations.

    act1: numpy array, (n_samples, n_features) from the first distribution
    act2: numpy array, (n_samples, n_features) from the second distribution
    """

    # 1. Calculate Mean and Covariance
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # 2. Calculate sum of squared differences between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # 3. Calculate the matrix square root of (sigma1 * sigma2)
    #    This is the trickiest part and why we use scipy
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # 4. Handle numerical instability:
    #    If the product matrix is singular, sqrtm can return complex numbers.
    #    Clip the imaginary part.
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 5. Calculate the trace (sum of diagonal elements)
    tr_covmean = np.trace(covmean)

    # 6. Final FD calculation
    fd = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fd


def compute_fpd_pca(real_images, generated_images, fitted_pca = None, n_components=0.85):
    """
    Computes the Fréchet PCA Distance (FPD) between two sets of images.

    real_images: numpy array, (n_samples, H, W) or (n_samples, H, W, C). Expected to be in range [-1, 1] or [0, 255]
    generated_images: numpy array, (n_samples, H, W) or (n_samples, H, W, C) . Expected to be in range [-1, 1] or [0, 255]
    n_components: int, number of principal components to use or float (0 < n_components < 1) for variance ratio threshold.
    """
    print(f"Calculating FPD with {n_components} PCA components...")

    # 1. Pre-process and flatten images
    #    - Reshape to 1D vectors

    real_min, real_max = real_images.min(), real_images.max()
    gen_min, gen_max = generated_images.min(), generated_images.max()

    print(f"Real images range: [{real_min:.4f}, {real_max:.4f}]")
    print(f"Generated images range: [{gen_min:.4f}, {gen_max:.4f}]")

    is_normalized = (real_min >= -1 and real_max <= 1 and
                     gen_min >= -1 and gen_max <= 1)

    if not is_normalized:
        print("Rescaling images to [0, 1] range...")
        real_images = (real_images - real_min) / (real_max - real_min)
        generated_images = (generated_images - gen_min) / (gen_max - gen_min)

    real_flat = real_images.reshape(real_images.shape[0], -1).astype(np.float32)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1).astype(np.float32)

    # 2. Fit PCA on REAL data
    if fitted_pca is None:
        print("Fitting PCA on real images...")
        pca = PCA(n_components=n_components) if fitted_pca is None else fitted_pca
        pca.fit(real_flat)
    else:
        pca = fitted_pca

    print(f'Total number of components used after PCA : {pca.n_components_}')

    # 3. Transform both sets into the PCA feature space
    real_features = pca.transform(real_flat)
    gen_features = pca.transform(gen_flat)

    # 4. Calculate Fréchet distance on the PCA features
    fpd = calculate_frechet_distance(real_features, gen_features)

    return fpd


# --- DEMO ---
if __name__ == "__main__":
    from torchvision.datasets import MNIST

    # 1. Load MNIST data
    mnist_dataset = MNIST(root='./data', train=True, download=True)
    x_train = mnist_dataset.data.numpy()  # Convert to numpy array
    x_test = MNIST(root='./data', train=False, download=True).data.numpy()

    # 2. Create our "datasets"
    N_SAMPLES = 10000

    # Set 1: Real images (our baseline)
    real_set_1 = x_train[:N_SAMPLES]

    # Set 2: "Perfect" generated images (just other real images)
    # This should result in a very low FPD score.
    real_set_2 = x_train[N_SAMPLES: N_SAMPLES * 2]

    # Set 3: "Mediocre" generated images (real images + some noise)
    noise = np.random.normal(0, 50, real_set_1.shape)
    noisy_set = (real_set_1 + noise).clip(0, 255)

    # Set 4: "Bad" generated images (pure random noise)
    # This should result in a very high FPD score.
    random_set = np.random.randint(0, 255, size=real_set_1.shape)

    # 3. Run the calculations

    print("--- Comparing Real vs. Real (should be near 0) ---")
    fpd_real_vs_real = compute_fpd_pca(real_set_1, real_set_2)
    print(f"FPD Score: {fpd_real_vs_real}\n")

    print("--- Comparing Real vs. Noisy (should be higher) ---")
    fpd_real_vs_noisy = compute_fpd_pca(real_set_1, noisy_set)
    print(f"FPD Score: {fpd_real_vs_noisy}\n")

    print("--- Comparing Real vs. Random (should be very high) ---")
    fpd_real_vs_random = compute_fpd_pca(real_set_1, random_set)
    print(f"FPD Score: {fpd_real_vs_random}\n")

    print("--- Comparing train vs. test (sanity check) ---")
    fpd_train_vs_test = compute_fpd_pca(real_set_1, x_test)
    print(f"FPD Score: {fpd_train_vs_test}\n")