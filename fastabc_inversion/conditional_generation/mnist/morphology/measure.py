"""
Morpho-MNIST
Adapted from https://github.com/dccastro/Morpho-MNIST/blob/main/morphomnist/measure.py
Adapted and extended by Eliane Maalouf for FastABC-Inversion
"""
import multiprocessing
from typing import NamedTuple

import numpy as np
import pandas as pd

from fastabc_inversion.conditional_generation.mnist.morphology.morpho import (
    bounding_parallelogram,
    ImageMoments,
    ImageMorphology,
)


class Morphometrics(NamedTuple):
    """Measured shape attributes of an image.

    - area: Total area/image mass.
    - length: Length of the estimated skeleton.
    - thickness: Mean thickness along the skeleton.
    - slant: Horizontal shear, in radians.
    - width: Width of the bounding parallelogram.
    - height: Height of the bounding parallelogram.
    """

    area: float
    length: float
    thickness: float
    slant: float
    width: float
    height: float


def measure_image(
    image,
    threshold: float = 0.5,
    scale: int = 4,
    bound_frac: float = 0.02,
    verbose=True,
):
    """Computes morphometrics for a single image.

    Parameters
    ----------
    image : (H, W) array_like
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
        between its minimum and maximum values.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis (>=1).
    bound_frac : float, optional
        Fraction of image mass to discard along each dimension when computing the bounding
        parallelogram.
    verbose : bool, optional
        Whether to pretty-print the estimated morphometrics.

    Returns
    -------
    Morphometrics
        A namedtuple containing the measured area, length, thickness, slant, width, and height.
    """
    image = np.asarray(image)
    morph = ImageMorphology(image, threshold, scale)
    moments = ImageMoments(morph.hires_image)
    thickness = morph.mean_thickness
    area = morph.area
    length = morph.stroke_length
    slant = np.arctan(-moments.horizontal_shear)
    slant = np.rad2deg(
        slant
    )  # added by Eliane Maalouf to convert to degrees by default

    corners = bounding_parallelogram(morph.hires_image, bound_frac, moments)
    width = (corners[1][0] - corners[0][0]) / morph.scale
    height = (corners[-1][1] - corners[0][1]) / morph.scale

    if verbose:
        print(f"Area: {area:.1f}")
        print(f"Length: {length:.1f}")
        print(f"Thickness: {thickness:.2f}")
        # print(f"Slant: {np.rad2deg(slant):.0f}°")
        print(
            f"Slant: {slant:.0f}°"
        )  # modified by Eliane Maalouf to print slant in degrees
        print(f"Dimensions: {width:.1f} x {height:.1f}")

    return Morphometrics(area, length, thickness, slant, width, height)


def _measure_image_unpack(arg):
    return measure_image(*arg)


def measure_batch(
    images,
    threshold: float = 0.5,
    scale: int = 4,
    bound_frac: float = 0.02,
    pool: multiprocessing.Pool = None,
    chunksize: int = 100,
) -> pd.DataFrame:
    """Computes morphometrics for a batch of images.

    Parameters
    ----------
    images : (N, H, W) array_like
        Input image batch, indexed along the first dimension.
    threshold : float, optional
        A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
        between its minimum and maximum values.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis (>1).
    bound_frac : float, optional
        Fraction of image mass to discard along each dimension when computing the bounding
        parallelogram.
    pool : multiprocessing.Pool, optional
        A pool of worker processes for parallel processing. Defaults to sequential computation.
    chunksize : int
        Size of the chunks in which to split the batch for parallel processing. Ignored if
        `pool=None`.

    Returns
    -------
    pandas.DataFrame
        A data frame with one row for each image, containing the following columns:

        - `area`: Total area/image mass.
        - `length`: Length of the estimated skeleton.
        - `thickness`: Mean thickness along the skeleton.
        - `slant`: Horizontal shear, in radians.
        - `width`: Width of the bounding parallelogram.
        - `height`: Height of the bounding parallelogram.

    Notes
    -----
    If the `tqdm` package is installed, this function will display a fancy progress bar with ETA.
    Otherwise, it will print a plain text progress message.
    """
    images = np.asarray(images)
    args = ((img, threshold, scale, bound_frac, False) for img in images)
    if pool is None:
        gen = map(_measure_image_unpack, args)
    else:
        gen = pool.imap(_measure_image_unpack, args, chunksize=chunksize)

    try:
        import tqdm

        gen = tqdm.tqdm(gen, total=len(images), unit="img", ascii=True)
    except ImportError:

        def plain_progress(g):
            print(f"\rProcessing images: {0}/{len(images)}", end="")
            for i, res in enumerate(g):
                print(f"\rProcessing images: {i + 1}/{len(images)}", end="")
                yield res
            print()

        gen = plain_progress(gen)

    results = list(gen)
    df = pd.DataFrame(results)
    return df


## added by Eliane Maalouf ####


def measure_slant(image, threshold: float = 0.5, scale: int = 4):
    """Computes only the slant for a single image.

    Parameters
    ----------
    image : (H, W) array_like
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis.

    Returns
    -------
    float
        Slant angle in radians.
    """
    image = np.asarray(image)
    morph = ImageMorphology(image, threshold, scale)
    moments = ImageMoments(morph.hires_image)
    slant = np.arctan(-moments.horizontal_shear)
    return slant


def measure_thickness(image, threshold: float = 0.5, scale: int = 4):
    """Computes only the thickness for a single image.

    Parameters
    ----------
    image : (H, W) array_like
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis.

    Returns
    -------
    float
        Mean thickness along the skeleton.
    """
    image = np.asarray(image)
    morph = ImageMorphology(image, threshold, scale)
    thickness = morph.mean_thickness
    return thickness


def measure_length(image, threshold: float = 0.5, scale: int = 4):
    """Computes only the length for a single image.

    Parameters
    ----------
    image : (H, W) array_like
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis.

    Returns
    -------
    float
        Length of the estimated skeleton.
    """
    image = np.asarray(image)
    morph = ImageMorphology(image, threshold, scale)
    length = morph.stroke_length
    return length


def distribution_measure(images, labels):
    """Computes the distribution of a specific morphometric over a batch of images.
    Plot a violin plot of the distribution per digit (0-9) and overall.

    Parameters
    ----------
    images : (N, H, W) array_like
        Input image batch, indexed along the first dimension.
    labels : (N,) array_like
        Labels corresponding to the images.
    Returns
    -------
    """

    import pandas as pd

    images = np.asarray(images)
    labels = np.asarray(labels)

    pool = multiprocessing.Pool()
    df = measure_batch(images, pool=pool)
    pool.close()
    pool.join()

    df["label"] = labels

    return df


def test_slant_on_mnist(num_samples=10):
    """Test slant measurement on MNIST dataset.

    Parameters
    ----------
    num_samples : int
        Number of images to test.
    """
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    # Load MNIST dataset (32x32 resized)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )

    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Calculate grid dimensions
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Test on first num_samples images
    slants = []
    for i in range(num_samples):
        image, label = mnist[i]
        # Convert from (C, H, W) tensor to (H, W) numpy array
        image_np = image.squeeze().numpy()

        slant = measure_slant(image_np)
        slant_degrees = np.rad2deg(slant)

        print(
            f"Image {i} (digit {label}): Slant = {slant:.4f} rad ({slant_degrees:.1f}°)"
        )
        slants.append(slant)

        # Plot the image
        axes[i].imshow(image_np, cmap="gray")
        axes[i].set_title(f"Digit {label}\nSlant: {slant_degrees:.1f}°")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("mnist_slant_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"\nMean slant: {np.mean(slants):.4f} rad ({np.rad2deg(np.mean(slants)):.1f}°)"
    )
    print(f"Std slant: {np.std(slants):.4f} rad ({np.rad2deg(np.std(slants)):.1f}°)")

    return slants


def test_thickness_on_mnist(num_samples=10):
    """Test thickness measurement on MNIST dataset.

    Parameters
    ----------
    num_samples : int
        Number of images to test.
    """

    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    # Load MNIST dataset (32x32 resized)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )

    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Calculate grid dimensions
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Test on first num_samples images
    thicknesses = []
    for i in range(num_samples):
        image, label = mnist[i]
        # Convert from (C, H, W) tensor to (H, W) numpy array
        image_np = image.squeeze().numpy()

        thickness = measure_thickness(image_np)

        print(f"Image {i} (digit {label}): Thickness = {thickness:.2f}")
        thicknesses.append(thickness)

        # Plot the image
        axes[i].imshow(image_np, cmap="gray")
        axes[i].set_title(f"Digit {label}\nThickness: {thickness:.2f}")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("mnist_thickness_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nMean thickness: {np.mean(thicknesses):.2f}")
    print(f"Std thickness: {np.std(thicknesses):.2f}")

    return thicknesses


# test distribution measurement for slant with MNIST torch dataset
def test_distribution_measure(num_samples=1000):
    """Test slant distribution measurement on MNIST dataset.

    Parameters
    ----------
    num_samples : int
        Number of images to test.
    """
    from torchvision import datasets, transforms

    # Load MNIST dataset (32x32 resized)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )

    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Collect all images and labels first
    images = []
    labels = []
    for i in range(num_samples):
        image, label = mnist[i]
        image_np = image.squeeze().numpy()
        images.append(image_np)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Process all images with distribution_measure
    df_morpho = distribution_measure(images, labels)

    return df_morpho


# Run the test
if __name__ == "__main__":
    # slants = test_slant_on_mnist(num_samples=20)
    # thicknesses = test_thickness_on_mnist(num_samples=20)
    create_df_morpho = False

    if create_df_morpho:
        df_morpho = test_distribution_measure(num_samples=60000)

        # pickle df_morpho
        import pickle

        with open("mnist_morphometrics.pkl", "wb") as f:
            pickle.dump(df_morpho, f)

    else:
        # load df_morpho from pickle
        import pickle

        with open("mnist_morphometrics.pkl", "rb") as f:
            df_morpho = pickle.load(f)

        mean_area = df_morpho["area"].mean()
        std_area = df_morpho["area"].std()
        range_area = (df_morpho["area"].min(), df_morpho["area"].max())
        median_area = df_morpho["area"].median()
        perc_25_area = df_morpho["area"].quantile(0.25)
        perc_75_area = df_morpho["area"].quantile(0.75)
        perc_2_5_area = df_morpho["area"].quantile(0.025)
        perc_97_5_area = df_morpho["area"].quantile(0.975)

        print(
            f"Area - Mean: {mean_area:.2f}, Std: {std_area:.2f}, Range: {range_area[0]:.2f} - {range_area[1]:.2f}, "
            f"Median: {median_area:.2f}, 25th Percentile: {perc_25_area:.2f}, 75th Percentile: {perc_75_area:.2f}, "
            f"2.5th Percentile: {perc_2_5_area:.2f}, 97.5th Percentile: {perc_97_5_area:.2f}"
        )

        mean_length = df_morpho["length"].mean()
        std_length = df_morpho["length"].std()
        range_length = (df_morpho["length"].min(), df_morpho["length"].max())
        median_length = df_morpho["length"].median()
        perc_25_length = df_morpho["length"].quantile(0.25)
        perc_75_length = df_morpho["length"].quantile(0.75)
        perc_2_5_length = df_morpho["length"].quantile(0.025)
        perc_97_5_length = df_morpho["length"].quantile(0.975)

        print(
            f"Length - Mean: {mean_length:.2f}, Std: {std_length:.2f}, Range: {range_length[0]:.2f} - {range_length[1]:.2f}, "
            f"Median: {median_length:.2f}, 25th Percentile: {perc_25_length:.2f}, 75th Percentile: {perc_75_length:.2f}, "
            f"2.5th Percentile: {perc_2_5_length:.2f}, 97.5th Percentile: {perc_97_5_length:.2f}"
        )

        mean_thickness = df_morpho["thickness"].mean()
        std_thickness = df_morpho["thickness"].std()
        range_thickness = (df_morpho["thickness"].min(), df_morpho["thickness"].max())
        median_thickness = df_morpho["thickness"].median()
        perc_25_thickness = df_morpho["thickness"].quantile(0.25)
        perc_75_thickness = df_morpho["thickness"].quantile(0.75)
        perc_2_5_thickness = df_morpho["thickness"].quantile(0.025)
        perc_97_5_thickness = df_morpho["thickness"].quantile(0.975)

        print(
            f"Thickness - Mean: {mean_thickness:.4f}, Std: {std_thickness:.4f}, Range: {range_thickness[0]:.4f} - {range_thickness[1]:.4f}, "
            f"Median: {median_thickness:.4f}, 25th Percentile: {perc_25_thickness:.4f}, 75th Percentile: {perc_75_thickness:.4f}, "
            f"2.5th Percentile: {perc_2_5_thickness:.4f}, 97.5th Percentile: {perc_97_5_thickness:.4f}"
        )

        mean_slant = df_morpho["slant"].mean()
        std_slant = df_morpho["slant"].std()
        range_slant = (df_morpho["slant"].min(), df_morpho["slant"].max())
        median_slant = df_morpho["slant"].median()
        perc_25_slant = df_morpho["slant"].quantile(0.25)
        perc_75_slant = df_morpho["slant"].quantile(0.75)
        perc_2_5_slant = df_morpho["slant"].quantile(0.025)
        perc_97_5_slant = df_morpho["slant"].quantile(0.975)

        print(
            f"Slant - Mean: {mean_slant:.2f}, Std: {std_slant:.2f}, Range: {range_slant[0]:.2f} - {range_slant[1]:.2f}, "
            f"Median: {median_slant:.2f}, 25th Percentile: {perc_25_slant:.2f}, 75th Percentile: {perc_75_slant:.2f}, "
            f"2.5th Percentile: {perc_2_5_slant:.2f}, 97.5th Percentile: {perc_97_5_slant:.2f}"
        )

        mean_width = df_morpho["width"].mean()
        std_width = df_morpho["width"].std()
        range_width = (df_morpho["width"].min(), df_morpho["width"].max())
        median_width = df_morpho["width"].median()
        perc_25_width = df_morpho["width"].quantile(0.25)
        perc_75_width = df_morpho["width"].quantile(0.75)
        perc_2_5_width = df_morpho["width"].quantile(0.025)
        perc_97_5_width = df_morpho["width"].quantile(0.975)

        print(
            f"Width - Mean: {mean_width:.2f}, Std: {std_width:.2f}, Range: {range_width[0]:.2f} - {range_width[1]:.2f}, "
            f"Median: {median_width:.2f}, 25th Percentile: {perc_25_width:.2f}, 75th Percentile: {perc_75_width:.2f}, "
            f"2.5th Percentile: {perc_2_5_width:.2f}, 97.5th Percentile: {perc_97_5_width:.2f}"
        )

        mean_height = df_morpho["height"].mean()
        std_height = df_morpho["height"].std()
        range_height = (df_morpho["height"].min(), df_morpho["height"].max())
        median_height = df_morpho["height"].median()
        perc_25_height = df_morpho["height"].quantile(0.25)
        perc_75_height = df_morpho["height"].quantile(0.75)
        perc_2_5_height = df_morpho["height"].quantile(0.025)
        perc_97_5_height = df_morpho["height"].quantile(0.975)

        print(
            f"Height - Mean: {mean_height:.2f}, Std: {std_height:.2f}, Range: {range_height[0]:.2f} - {range_height[1]:.2f}, "
            f"Median: {median_height:.2f}, 25th Percentile: {perc_25_height:.2f}, 75th Percentile: {perc_75_height:.2f}, "
            f"2.5th Percentile: {perc_2_5_height:.2f}, 97.5th Percentile: {perc_97_5_height:.2f}"
        )

        from fastabc_inversion.conditional_generation.utils.plotting import (
            plot_horizontal_boxplots_with_total,
        )

        # Prepare data for each metric
        df_slant = df_morpho[["label", "slant"]].rename(
            columns={"label": "labels", "slant": "values"}
        )
        df_thickness = df_morpho[["label", "thickness"]].rename(
            columns={"label": "labels", "thickness": "values"}
        )
        df_length = df_morpho[["label", "length"]].rename(
            columns={"label": "labels", "length": "values"}
        )

        # Plot slant
        plot_horizontal_boxplots_with_total(
            df_slant,
            title="Slant Distribution by Digit",
            save_location="slant_boxplots.pdf",
            dpi=600,
            show=False,
        )

        # Plot thickness
        plot_horizontal_boxplots_with_total(
            df_thickness,
            title="Thickness Distribution by Digit",
            save_location="thickness_boxplots.pdf",
            dpi=600,
            show=False,
        )

        # Plot length
        plot_horizontal_boxplots_with_total(
            df_length,
            title="Length Distribution by Digit",
            save_location="length_boxplots.pdf",
            dpi=600,
            show=False,
        )
