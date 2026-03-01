"""
Common plotting configurations and functions.
"""
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter


def plots_imports():
    import matplotlib as mpl

    mpl.use("module://backend_interagg")
    # mpl.use('pdf')  # choose pdf renderer for vector graphic # default was: 'module://backend_interagg'
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    return mpl, plt, make_axes_locatable, tick


def base_config(
    mpl=None,
    figsize=(8, 6),
    family="serif",
    fonttype="Liberation Serif",
    texfontType="dejavuserif",
    dpi=600,
    fontsize=14,
):
    """base_config
    Function to setup common figures configuration
    :param mpl: matplotlib module to use. If None, it will import matplotlib.
    :param figsize: specify figure size (width, height) in inches. Default: (8in , 6in)
    :param family: font family for text. Default: serif
    :param fonttype: specific font type in family. Default: Liberation Serif
    :param texfontType:  Tex (for math text) font type. Default: dejavuserif
    :param dpi: specify image resolution in dots per inch. Default: 600
    :param fontsize: font size in points. Default: 14 pt
    """
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["font.family"] = family
    mpl.rcParams["font.{}".format(family)] = fonttype
    mpl.rcParams["mathtext.fontset"] = texfontType
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["lines.linewidth"] = 1


def plot_histograms(
    data,
    labels,
    plot_title,
    weights=None,
    bins=30,
    colors=None,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot histograms of given data.
    :param data: array containing the data to plot. Format as [number of histograms, values per histogram]
    :param labels : labels to give to each histogram, should be a list with size "number of histograms"
    :param weights : weights to give to each element of the data in each histogram.
                    If None, all elements have equal weight = 1/(number of elements in histogram)
    :param bins: number of bins to use for the histogram
    :param colors: list of colors to use for each histogram, should be a list with size "number of histograms".
                  If None, colors are automatically assigned.
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    :return:
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    if colors is None:
        colors = sns.color_palette("muted", n_colors=data.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    weights = np.ones(data.shape[1]) / data.shape[1] if weights is None else weights

    for i in range(data.shape[0]):
        ax.hist(
            data[i, :],
            bins=bins,
            color=colors[i],
            edgecolor=colors[i],
            alpha=0.5,
            weights=weights,
            label=labels[i],
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.legend(fontsize="small")
    plt.title(plot_title)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_true_vs_recon_grid(
    samples,
    recon_rmse,
    labels,
    num_samples,
    cmap="gray",
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot a grid of true vs reconstructed images.
    :param samples: array containing the samples to plot.
        Format as [num_samples *2, height, width], first half are true images, second half are reconstructions
    :param recon_rmse: array containing the reconstruction RMSE for each sample. Format as [num_samples]
    :param labels: labels for each sample. Format as [num_samples * 2], first half are true labels, second half are reconstructions
    :param num_samples: number of samples being plotted, the grid will be of size sqrt(num_samples) x 2*sqrt(num_samples)
    :param save_location: location where to save the generated plot
    :param dpi:  resolution of the image in dots per inch (dpi)
    :param show: whether to show the plot or save it
    :param kwargs: additional arguments to pass to base_config
    :return:
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(
        nrows=grid_size, ncols=grid_size * 2, figsize=(grid_size * 2, grid_size)
    )

    vmin = np.min(samples)
    vmax = np.max(samples)

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            im_1 = axes[i, 2 * j].imshow(samples[index, :, :], cmap=cmap)
            axes[i, 2 * j].axis("off")
            axes[i, 2 * j].set_title(f"True, {labels[index]}", fontsize=8)
            im_1.set_clim(vmin=vmin, vmax=vmax)

            im_2 = axes[i, 2 * j + 1].imshow(
                samples[num_samples + index, :, :], cmap=cmap
            )
            axes[i, 2 * j + 1].axis("off")
            axes[i, 2 * j + 1].set_title(
                f"RMSE= {recon_rmse[index]:.4f}, {labels[num_samples + index]}",
                fontsize=8,
            )
            im_2.set_clim(vmin=vmin, vmax=vmax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_samples_grid(
    images,
    labels,
    num_samples=None,
    cmap="gray",
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot a grid of images.
    :param images: array containing the samples to plot.
        Format as [num_samples, height, width]
    :param labels: labels for each sample. Format as [num_samples]
    :param save_location: location where to save the generated plot
    :param dpi:  resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    num_samples = num_samples if num_samples is not None else images.shape[0]
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(
        nrows=grid_size, ncols=grid_size, figsize=(grid_size, grid_size)
    )

    vmin = np.min(images)
    vmax = np.max(images)

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            im = axes[i, j].imshow(images[index, :, :], cmap=cmap)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{labels[index]}", fontsize=8)
            im.set_clim(vmin=vmin, vmax=vmax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_cov(
    cov, plot_title, vmin_vmax=None, save_location=None, dpi=600, show=False, **kwargs
):
    """
    Function to plot a covariance matrix.
    :param cov: the covariance matrix to plot
    :param plot_title: the title to use for the plot
    :param vmin_vmax: tuple containing the minimum and maximum values to use for the colorbar. If None, the values are set
    to the minimum and maximum values of the covariance matrix.
    :param file_name_key: the key to use for the file name. If None, the key is set to 'covm'
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    :return:
    """
    vmin = vmin_vmax[0] if vmin_vmax is not None else np.min(cov)
    vmax = vmin_vmax[1] if vmin_vmax is not None else np.max(cov)

    dim = cov.shape[0]

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    im = ax.imshow(cov)
    im.set_clim(vmin, vmax)
    ax.set_xticks(range(dim), range(1, dim + 1))
    ax.set_yticks(range(dim), range(1, dim + 1))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    ax.title.set_text(plot_title)
    fig.colorbar(im, cax=cax)
    if show:
        plt.show()
    else:
        plt.savefig(f"{save_location}", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_scatters(
    data,
    labels,
    plot_title,
    make_scatter=True,
    axis_labels=None,
    colors=None,
    markers=None,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot scatter plots of given data.
    :param data: array containing the data to plot. Format as [number of scatter plots, x-axis values, y-axis values]
    :param labels : labels to give to each scatter plot, should be a list with size "number of scatter plots"
    :param plot_title: title to use for the plot
    :param make_scatter: if True, make scatter plot. If False, make line plot.
    :param axis_labels: should be a dictionary with keys 'x_label' and 'y_label' and values as the labels to use for the x and y axis.
    :param colors: list of colors to use for each scatter plot, should be a list with size "number of scatter plots".
                     if None, colors are automatically assigned.
    :param markers: list of markers to use for each scatter plot, should be a list with size "number of scatter plots".
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    colors = (
        sns.color_palette("muted", n_colors=data.shape[0]) if colors is None else colors
    )
    fig, ax = plt.subplots(figsize=(20, 20))

    for i in range(data.shape[0]):
        if make_scatter:
            ax.scatter(
                data[i, :, 0],
                data[i, :, 1],
                c=colors[i],
                label=labels[i],
                marker=markers[i] if markers is not None else ".",
            )
        else:
            ax.plot(data[i, :, 0], data[i, :, 1], c=colors[i], label=labels[i])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if axis_labels is not None:
        ax.set_xlabel(axis_labels["x_label"])
        ax.set_ylabel(axis_labels["y_label"])
    plt.legend(fontsize="small")
    plt.title(plot_title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_samples_inspections(
    images,
    labels,
    labels_2=None,
    random_samples=True,
    grid_size=5,
    save_fig_path=None,
    dpi=600,
    **kwargs,
):
    """
    Plot samples on a grid
    images: tensor of images
    labels: tensor of labels
    labels_2: optional second set of labels to display. Mostly used to show classifier assigned predictions.
    random_samples: if True, display random samples; if False, display grid_size samples per class
    grid_size: size of the grid to display samples (grid_size x grid_size) when random_samples is True
    save_fig_path: if provided, save the figure under the given path as a PDF. The path should include the filename.
                    If None, display the figure instead.
    dpi: resolution of the saved figure
    kwargs: additional keyword arguments for plotting configuration
    """
    # TODO : make independent from torch

    import random
    import torch
    from fastabc_inversion.conditional_generation.utils.plotting import (
        plots_imports,
        base_config,
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    if random_samples:
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )
        for i in range(grid_size):
            for j in range(grid_size):
                idx = random.randint(0, images.shape[0] - 1)
                axes[i, j].imshow(images[idx].squeeze(), cmap="gray")
                axes[i, j].set_title(f"Label: {labels[idx].item()}")
                axes[i, j].axis("off")
    else:
        num_classes = len(torch.unique(labels))
        fig, axes = plt.subplots(
            num_classes, grid_size, figsize=(grid_size * 2, num_classes * 2)
        )
        for c in range(num_classes):
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            for j in range(grid_size):
                if j < len(class_indices):
                    idx = class_indices[j].item()
                    axes[c, j].imshow(images[idx].squeeze(), cmap="gray")
                    if labels_2 is None:
                        axes[c, j].set_title(f"Label: {labels[idx].item()}")
                    else:
                        axes[c, j].set_title(
                            f"Label: {labels[idx].item()}, label_2: {labels_2[idx].item()}"
                        )
                axes[c, j].axis("off")

    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_stripplot(
    df,
    title,
    value_labels=None,
    yticks=None,
    reverse_labels=False,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Plots stripplots from a DataFrame with optional saving and customization features.

    The function generates stripplots based on 'labels' and 'values' columns from the given
    DataFrame. Optionally, it allows saving the generated plot to a specified location
    and customizing the visual appearance of the plot. Additionally, if the 'values'
    in the dataset are lists or NumPy arrays, the function explodes these into individual
    rows for accurate plotting.

    :param df: Input DataFrame with at least two columns: 'labels' and 'values'.
    :param title: Title of the plot.
    :param value_labels: List of labels for each unique value category (for legend). If None, no legend is shown.
    :param yticks: List of tick positions to show on the y-axis. If None, uses default ticks.
    :param reverse_labels: If True, reverses the order of labels on the y-axis (highest to lowest).
    :param save_location: File path to save the plot. If None, the plot is not saved.
    :param dpi: Resolution in dots per inch for saving the plot. Default is 600.
    :param show: If True, displays the plot. If False, the plot is saved without being displayed.
    :param kwargs: Additional keyword arguments for customizing the plot.
    :return: None
    """

    import pandas as pd
    from fastabc_inversion.conditional_generation.utils.plotting import (
        plots_imports,
        base_config,
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    fig, ax = plt.subplots()

    # Determine order for x-axis
    order = df["labels"].unique()[::-1] if reverse_labels else None

    # Check if 'class' column exists for color coding
    use_class_colors = "class" in df.columns

    # Set up color palette for classes (0-9)
    if use_class_colors:
        colors = sns.color_palette("muted", n_colors=10)
        palette = {i: colors[i] for i in range(10)}
    else:
        palette = None

    # Common stripplot configuration
    stripplot_config = {
        "jitter": True,
        "alpha": 0.6,
        "orient": "v",
        "linewidth": 0,
        "size": 3,
        "dodge": False,
        "order": order,
    }

    # Check if values are lists/arrays and explode if needed
    if isinstance(df["values"].iloc[0], (list, np.ndarray)):
        df_exploded = df.explode("values")
        df_exploded["values"] = pd.to_numeric(df_exploded["values"])

        # Create hue column based on original index position within each group
        df_exploded["value_category"] = df_exploded.groupby(level=0).cumcount()

        if value_labels is not None:
            df_exploded["value_category"] = df_exploded["value_category"].map(
                lambda x: value_labels[x] if x < len(value_labels) else f"Value {x}"
            )

            # Use 'class' for coloring if available, otherwise use value_category
            hue_column = "class" if use_class_colors else "value_category"

            # For the case with hue (multi-value)
            sns.stripplot(
                x="labels",
                y="values",
                data=df_exploded,
                ax=ax,
                hue=hue_column,
                palette=palette,
                **stripplot_config,
            )

    else:
        # For the simple case (single value)
        if use_class_colors:
            sns.stripplot(
                x="labels",
                y="values",
                data=df,
                ax=ax,
                hue="class",
                palette=palette,
                **stripplot_config,
            )
        else:
            sns.stripplot(
                x="labels",
                y="values",
                data=df,
                ax=ax,
                color="#1f77b4",
                **stripplot_config,
            )

    ax.set_title(title)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if yticks is not None:
        ax.set_yticks(yticks)

    # Show legend if value_labels provided OR if using class colors
    if value_labels is not None or use_class_colors:
        legend = ax.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
        # Apply same visual config to legend markers
        for handle in legend.legend_handles:
            handle.set_alpha(stripplot_config["alpha"])
            handle.set_sizes([stripplot_config["size"] ** 2])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_boxplot_with_stripplot(
    df,
    title,
    value_labels=None,
    yticks=None,
    reverse_labels=False,
    sample_frac=0.2,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Plots boxplots with overlaid stripplots from a DataFrame with optional saving and customization features.

    The function generates boxplots with overlaid stripplots based on 'labels' and 'values' columns from the given
    DataFrame. The boxplot uses all data points, while the stripplot shows only a random subset.

    :param df: Input DataFrame with at least two columns: 'labels' and 'values'.
    :param title: Title of the plot.
    :param value_labels: List of labels for each unique value category (for legend). If None, no legend is shown.
    :param yticks: List of tick positions to show on the y-axis. If None, uses default ticks.
    :param reverse_labels: If True, reverses the order of labels on the y-axis (highest to lowest).
    :param sample_frac: Fraction of points to show in stripplot (between 0 and 1). Default is 0.5 (50%).
    :param save_location: File path to save the plot. If None, the plot is not saved.
    :param dpi: Resolution in dots per inch for saving the plot. Default is 600.
    :param show: If True, displays the plot. If False, the plot is saved without being displayed.
    :param kwargs: Additional keyword arguments for customizing the plot.
    :return: None
    """

    import pandas as pd
    from fastabc_inversion.conditional_generation.utils.plotting import (
        plots_imports,
        base_config,
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    fig, ax = plt.subplots()

    # Determine order for x-axis
    order = df["labels"].unique()[::-1] if reverse_labels else None

    # Check if 'class' column exists for color coding
    use_class_colors = "class" in df.columns

    # Set up color palette for classes (0-9)
    if use_class_colors:
        colors = sns.color_palette("muted", n_colors=10)
        palette = {i: colors[i] for i in range(10)}
    else:
        palette = None

    # Boxplot configuration
    boxplot_config = {
        "color": "lightsteelblue",
        "linewidth": 1.0,
        "fliersize": 0,
        "showcaps": True,
        "boxprops": dict(alpha=0.4),
        "order": order,
        "width": 0.4,
        "whis": [2.5, 97.5],
    }

    # Common stripplot configuration
    stripplot_config = {
        "jitter": True,
        "alpha": 0.6,
        "orient": "v",
        "linewidth": 0,
        "size": 3,
        "dodge": False,
        "order": order,
    }

    # Check if values are lists/arrays and explode if needed
    if isinstance(df["values"].iloc[0], (list, np.ndarray)):
        df_exploded = df.explode("values")
        df_exploded["values"] = pd.to_numeric(df_exploded["values"])

        # Create hue column based on original index position within each group
        df_exploded["value_category"] = df_exploded.groupby(level=0).cumcount()

        if value_labels is not None:
            df_exploded["value_category"] = df_exploded["value_category"].map(
                lambda x: value_labels[x] if x < len(value_labels) else f"Value {x}"
            )

            # Use 'class' for coloring if available, otherwise use value_category
            hue_column = "class" if use_class_colors else "value_category"

            # Plot boxplot first with ALL data
            sns.boxplot(
                x="labels", y="values", data=df_exploded, ax=ax, **boxplot_config
            )

            # Sample a random subset for stripplot
            df_sampled = df_exploded.sample(frac=sample_frac, random_state=42)

            # Plot stripplot on top with sampled data
            sns.stripplot(
                x="labels",
                y="values",
                data=df_sampled,
                ax=ax,
                hue=hue_column,
                palette=palette,
                **stripplot_config,
            )

    else:
        # For the simple case (single value)
        # Plot boxplot first with ALL data
        sns.boxplot(x="labels", y="values", data=df, ax=ax, **boxplot_config)

        # Sample a random subset for stripplot
        df_sampled = df.sample(frac=sample_frac, random_state=42)

        # Plot stripplot on top with sampled data
        if use_class_colors:
            sns.stripplot(
                x="labels",
                y="values",
                data=df_sampled,
                ax=ax,
                hue="class",
                palette=palette,
                **stripplot_config,
            )
        else:
            sns.stripplot(
                x="labels",
                y="values",
                data=df_sampled,
                ax=ax,
                color="#1f77b4",
                **stripplot_config,
            )

    ax.set_title(title)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Dynamic yticks handling
    if yticks is not None:
        # Get the actual data range
        df_values = (
            df_exploded["values"]
            if isinstance(df["values"].iloc[0], (list, np.ndarray))
            else df["values"]
        )
        max_value = df_values.max()
        max_ytick = max(yticks)

        # If data exceeds provided yticks, extend them
        if max_value > max_ytick:
            tick_step = yticks[1] - yticks[0] if len(yticks) > 1 else 1
            extended_yticks = list(yticks)
            while extended_yticks[-1] < max_value:
                extended_yticks.append(extended_yticks[-1] + tick_step)
            ax.set_yticks(extended_yticks)
        else:
            ax.set_yticks(yticks)

    # Show legend if value_labels provided OR if using class colors
    if value_labels is not None or use_class_colors:
        legend = ax.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
        # Apply same visual config to legend markers
        for handle in legend.legend_handles:
            handle.set_alpha(stripplot_config["alpha"])
            handle.set_sizes([stripplot_config["size"] ** 2])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_horizontal_boxplots_with_total(
    df, title, save_location=None, dpi=600, show=False, **kwargs
):
    """
    Plots horizontal boxplots for each class (0-9) plus an overall boxplot for all measurements.

    :param df: Input DataFrame with columns 'labels' (class 0-9) and 'values' (measurements).
    :param title: Title of the plot.
    :param save_location: File path to save the plot. If None, the plot is not saved.
    :param dpi: Resolution in dots per inch for saving the plot. Default is 600.
    :param show: If True, displays the plot. If False, the plot is saved without being displayed.
    :param kwargs: Additional keyword arguments for customizing the plot.
    :return: None
    """
    import pandas as pd
    from fastabc_inversion.conditional_generation.utils.plotting import (
        plots_imports,
        base_config,
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Add vertical grid lines first (lower zorder)
    ax.grid(axis="x", alpha=0.4, linestyle="-", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Prepare data with "All" category
    df_with_all = df.copy()
    df_all = df.copy()
    df_all["labels"] = "All"
    df_combined = pd.concat([df_with_all, df_all], ignore_index=True)

    # Define order: 9 to 0, then All at bottom
    order = [str(i) for i in range(9, -1, -1)] + ["All"]
    df_combined["labels"] = df_combined["labels"].astype(str)

    # Set up color palette matching boxplot_with_stripplot
    colors = sns.color_palette("muted", n_colors=10)
    palette = {str(i): colors[i] for i in range(10)}
    palette["All"] = "lightsteelblue"

    # Boxplot configuration with lighter grey
    boxplot_config = {
        "linewidth": 1.2,
        "fliersize": 0,
        "showcaps": True,
        "boxprops": dict(alpha=0.7, edgecolor="grey"),
        "whiskerprops": dict(color="grey"),
        "capprops": dict(color="grey"),
        "medianprops": dict(color="grey"),
        "order": order,
        "orient": "h",
        "whis": [2.5, 97.5],
        "palette": palette,
        "zorder": 2,
    }

    # Create horizontal boxplot
    sns.boxplot(y="labels", x="values", data=df_combined, ax=ax, **boxplot_config)

    # Add diamond markers for min/max values with smaller size
    for i, label in enumerate(order):
        label_data = df_combined[df_combined["labels"] == label]["values"]
        min_val = label_data.min()
        max_val = label_data.max()
        ax.scatter([min_val, max_val], [i, i], marker="D", s=10, color="grey", zorder=3)

    # Set x-axis to data range
    x_min = df["values"].min()
    x_max = df["values"].max()
    x_margin = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - x_margin, x_max + x_margin)

    ax.set_title(title)
    ax.set_ylabel("Class")
    ax.set_xlabel("Measurement")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_class_proportions_stacked(
    df,
    threshold_limit=None,
    title=None,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Plots stacked bar chart showing class proportions at different labels/thresholds.

    :param df: DataFrame with columns 'labels' (threshold/category) and 'values' (predicted classes 0-9)
    :param threshold_limit: Optional upper limit for thresholds to plot. Only thresholds <= this value will be shown.
    :param title: Title of the plot
    :param save_location: File path to save the plot
    :param dpi: Resolution in dots per inch
    :param show: If True, displays the plot
    :param kwargs: Additional keyword arguments for customizing the plot
    """
    from fastabc_inversion.conditional_generation.utils.plotting import (
        plots_imports,
        base_config,
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)

    # Filter data by threshold limit if provided
    if threshold_limit is not None:
        df = df[df["labels"] <= threshold_limit]

    # Count occurrences of each class at each threshold
    counts = df.groupby(["labels", "values"]).size().reset_index(name="count")

    # Pivot data for stacking
    pivot_df = counts.pivot_table(
        index="labels", columns="values", values="count", fill_value=0
    )
    pivot_df = (
        pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    )  # Convert to percentages

    # Reverse the order of labels (highest to lowest)
    pivot_df = pivot_df.iloc[::-1]

    # Set up colors for classes 0-9
    colors = sns.color_palette("muted", n_colors=10)

    # Calculate dynamic bar width based on number of labels (narrower bars, avoid jamming)
    num_labels = len(pivot_df)
    bar_width = max(
        0.3, min(0.7, 5.0 / num_labels)
    )  # Between 0.3 and 0.7, scaled by count

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df.plot(
        kind="bar", stacked=True, ax=ax, color=colors, width=bar_width, alpha=0.6
    )

    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("Threshold")
    ax.set_title(title if title else "Class Proportions at Different Thresholds")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    ax.yaxis.set_major_formatter(PercentFormatter())

    # Match visual config from plot_boxplot_with_stripplot
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=0)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()
