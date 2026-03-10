"""
STATISTICS:
-----------

Module containinig methods to plot summary statistics of given dataset
"""
# resource : https://towardsdatascience.com/making-publication-quality-figures-in-python-part-i-fig-and-axes-d86c3903ad9b
import matplotlib as mpl

mpl.use("module://backend_interagg")
import pickle
from math import sqrt

import gstools as gs
import h5py
import matplotlib.gridspec as gridspec
# mpl.use('pdf')  # choose pdf renderer for vector graphic # default was: 'module://backend_interagg'
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fastabc_inversion.geo_problems.utils.config import Config
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def plot_example(
    example_x,
    example_y=None,
    noisy_example_y=None,
    save_location=None,
    dpi=600,
    **kwargs,
):
    """plot_example
    Function to plot a given example.
    :param example_x: the x part of the example (i.e., the subsruface domain)
    :param example_y: the y part of the example (i.e., the measurement vector)
    :param noisy_example_y: if not None, plots the measurement vector with noise (i.e., with measurement uncertainty)
    :param save_location: folder location where to save the resulting plot
    :param dpi: the image resolution in dots per inch. Default: 600
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig = plt.figure()
    gspec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[30, 60])

    # plot x (image)
    ax_x = plt.subplot(gspec[0, 0], aspect=1.25)
    ax_x.set_xlabel(r"Width (m)")
    ax_x.set_ylabel(r"Depth (m)")
    ax_x.set_yticks([-0.5, 10, 20, 30, 40, 49.5])
    ax_x.set_yticklabels([0, 1, 2, 3, 4, 5])
    ax_x.set_xticks([-0.5, 10, 20, 30, 39.5])
    ax_x.set_xticklabels([0, 1, 2, 3, 4])
    ax_x.title.set_text(r"Slowness field (ns/m)")
    im_x = ax_x.imshow(example_x)

    divider = make_axes_locatable(ax_x)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_x, cax=cax, format=tick.FormatStrFormatter("%.1f"))

    # plot y (line)
    ax_y = plt.subplot(gspec[0, 1], aspect=1.95)
    ax_y.set_xlabel("Ray number")
    ax_y.set_ylabel("Travel times (ns)")
    ax_y.spines["right"].set_visible(False)
    ax_y.spines["top"].set_visible(False)
    ax_y.spines["left"].set_position(("outward", 5))
    ax_y.spines["left"].set_linewidth(0.5)
    ax_y.spines["bottom"].set_linewidth(0.5)
    ax_y.set_xlim((0, len(example_y) - 1))
    ax_y.title.set_text(r"First arrival travel times (ns)")

    if example_y is not None:
        ax_y.plot(example_y, linestyle="solid", label="Without noise")

    if noisy_example_y is not None:
        ax_y.plot(noisy_example_y, linestyle="dashed", c="black", label="With noise")

    if example_y is None and noisy_example_y is None:
        raise ValueError(
            "At least one of the two arguments example_y and noisy_example_y must be provided."
        )

    ax_y.legend(edgecolor="None")

    plt.tight_layout()

    # plt.show()
    plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_resimulations(resimulations, ref_y, save_location=None, dpi=600, **kwargs):
    """
    Function to plot multiple travel times against each other. Mainly, to plot the noiseless travel times, the noisy
    travel times, and the resimulations of the posterior samples through the forward solver.
    :param resimulations: a numpy array containing the travel times to plot. Should be formated as [number of lines to plot, number of rays].
    :param ref_y: a numpy array containing the reference. Should be formated as [2, number of rays].
    The first rays should be the noiseless ones. The second rays should be the noisy ones.
    :param save_location:  folder location where to save the resulting plot
    :param dpi: the image resolution in dots per inch. Default: 600
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig, ax = plt.subplots(nrows=1, ncols=1)

    lines_to_plot = resimulations.shape[0]

    ax.plot(ref_y[0], linestyle="solid")  # noiseless
    ax.plot(ref_y[1], linestyle="dashed", c="black")  # noisy measurement

    for i in range(lines_to_plot):
        ax.plot(resimulations[i, :].reshape(-1), c="gray", alpha=0.4)  # resimulations

    ax.set_xlabel("Ray number")
    ax.set_ylabel("Travel times (ns)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xlim((0, len(ref_y[0]) - 1))

    plt.tight_layout()

    # plt.show()
    plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_summary_models(
    samples,
    width,
    height,
    save_location=None,
    dpi=600,
    gen_plots=True,
    plot_var=False,
    **kwargs,
):
    """plot_summary_models
    Function to generate pixel wise mean, variance and std of a set of models.
    The function also plots the mean and the standard deviation. Variance is plotted if plot_var = True
    :param samples: the dataset for which to calculate (and plot) pixel wise statistics. Expects data in format [number of samples, width*height]
    :param width: the width of the image in pixels
    :param height: the height of the image in pixels
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param gen_plots: if True, generate plots for the mean and standard deviation (optional: variance)
    :param plot_var: plot pixel wise variance along the mean and the standard deviation
    :param kwargs: additional arguments to pass to base_config
    """
    mean_vector = np.mean(samples, axis=0)  # pixel wise mean over all samples
    std_vector = np.std(
        samples, axis=0
    )  # pixel wise standard deviation over all samples
    var_vector = std_vector**2

    # plots
    if gen_plots:
        mpl, plt, make_axes_locatable, tick = plots_imports()
        base_config(mpl, **kwargs)
        fig = plt.figure()

        if plot_var:
            cols = 3
            all_vectors = np.array([mean_vector, var_vector, std_vector]).reshape(
                cols, width * height
            )
        else:
            cols = 2
            all_vectors = np.array([mean_vector, std_vector]).reshape(
                cols, width * height
            )

        gspec = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)

        for i in range(cols):
            ax = fig.add_subplot(gspec[0, i])

            im = ax.imshow(all_vectors[i, :].reshape(height, width))
            ax.set_xticks([])
            ax.set_yticks([])
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

            if i == 0:
                ax.title.set_text(r"Mean (ns/m)")
            if i == 1 and cols == 3:
                ax.title.set_text(r"Variance (ns/m)²")
            elif i == 1 and cols == 2:
                ax.title.set_text(r"Standard deviation (ns/m)")
            else:
                pass

        plt.tight_layout()

        # plt.show()
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

        plt.close()

        return (mean_vector, std_vector)

    else:
        return (mean_vector, std_vector)


def plot_prior_vs_posterior_stats(
    prior_stats,
    posterior_stats,
    width,
    height,
    save_location=None,
    dpi=600,
    left_title=r"Prior {}",
    right_title=r"Posterior {}",
    **kwargs,
):
    """
    Function to plot prior statistics (mean, std) against posterior statistics
    :param prior_stats: Tuple of mean and std of prior distribution
    :param posetrior_stats: Tuple of mean and std of posterior distribution
    :param width: the width of the image in pixels
    :param height: the height of the image in pixels
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param left_title: title to use for the left plot. Default: "Prior {}"
    :param right_title: title to use for the right plot. Default: "Posterior {}"
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig = plt.figure()

    rows = 2
    cols = 2

    gspec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)

    all_means = np.array([prior_stats[0], posterior_stats[0]]).reshape(
        cols, width * height
    )
    all_std = np.array([prior_stats[1], posterior_stats[1]]).reshape(
        cols, width * height
    )

    for i in range(rows):
        if i == 0:
            name = "mean (ns/m)"
            vec = all_means
        else:
            name = "standard deviation (ns/m)"
            vec = all_std

        vmin = np.min(vec)
        vmax = np.max(vec)

        for j in range(cols):
            ax = fig.add_subplot(gspec[i, j])

            im = ax.imshow(vec[j, :].reshape(height, width))
            ax.set_xticks([])
            ax.set_yticks([])
            im.set_clim(vmin, vmax)
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
            cax.set_xticks([])
            cax.set_yticks([])

            if j == cols - 1:
                fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

            if j == 0:
                ax.title.set_text(left_title.format(name))
            else:
                ax.title.set_text(right_title.format(name))

    plt.tight_layout()

    # plt.show()
    plt.savefig(save_location, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_samples(
    examples,
    width,
    height,
    rmse_labels=None,
    ssim_labels=None,
    grd_truth=True,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot given set of examples
    :param examples: a set of examples to plot. Expects format as (number of examples, samples per example, height*width)
    :param width: the width of the image in pixels
    :param height: the height of the image in pixels
    :param rmse_labels: a list of RMSE values to add on top of the sample
    :param ssim_labels: a list of SSIM values to add on top of the sample
    :param grd_truth: if True, the first example provided is the ground truth
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig = plt.figure()

    rows = examples.shape[0]
    cols = examples.shape[1]

    gspec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)

    vmin = np.min(examples)
    vmax = np.max(examples)

    for i in range(rows):
        # vmin = np.min(examples[i, :, :])
        # vmax = np.max(examples[i, :, :])

        for j in range(cols):
            ax = fig.add_subplot(gspec[i, j])

            im = ax.imshow(examples[i, j, :].reshape(height, width))
            ax.set_xticks([])
            ax.set_yticks([])
            im.set_clim(vmin, vmax)
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
            cax.set_xticks([])
            cax.set_yticks([])

            if j == cols - 1:
                fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

            if grd_truth:
                if j == 0:
                    ax.title.set_text(r"Ground truth")
                else:
                    label = (
                        f"RMSE = {rmse_labels[i][j-1]:.2f} ns/m"
                        if rmse_labels is not None
                        else f"Example #{j}"
                    )
                    label = (
                        f"{label}; SSIM = {ssim_labels[i][j-1]:.2f}"
                        if ssim_labels is not None
                        else label
                    )
                    ax.title.set_text(label)
            else:
                label = (
                    f"RMSE = {rmse_labels[i][j]:.2f} ns/m"
                    if rmse_labels is not None
                    else f"Example #{j}"
                )
                label = (
                    f"{label}; SSIM = {ssim_labels[i][j]:.2f}"
                    if ssim_labels is not None
                    else label
                )
                ax.title.set_text(label)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_boxplots(
    values_all,
    labels,
    references_dict=None,
    axes_plot_titles=None,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot boxplots of given values. It detects if one or more boxplots need to be drawn and assigns labels
    to them.
    :param values_all: array containing the values to boxplot. Format as [number of subplots, number of boxplots, values per boxplot]
    to have multiple sublopts in the same figure and multiple boxplots in each subplot.
    :param labels : labels to give to each boxplot, should be a list with size "number of boxplots"
    :param references_dict: dictionary containing the reference values for the boxplots. It should be structured as follows:
        {'ref1': {'lower': value, 'center': value, 'upper': value},
        'ref2': {'lower': value, 'center': value, 'upper': value}}.
        For relevance of interpretation, the center value should represent a median and lower-upper should represent
        IQR bounds.
    :param axes_plot_titles : provide title to use for axes and plot. Format as [list of subplot_titles, horizontal_axis_title, vertical_axis_title]
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param show: if True, the plot is shown but not saved. if False, it is only saved
    :param kwargs: configure plotting parameters. For example, 'whis_low', 'whis_high', 'lower_lim', 'upper_lim', 'h_axis_margin',
        'v_axis_margin', 'x_ticks_step', 'y_scale'.
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl)

    # get kwargs if any or set default values
    whis_low = kwargs.get("whis_low", 2.5)
    whis_high = kwargs.get("whis_high", 97.5)
    lower_lim = kwargs.get("lower_lim", None)
    upper_lim = kwargs.get("upper_lim", None)
    h_axis_margin = kwargs.get("h_axis_margin", 0.7)
    v_axis_margin = kwargs.get("v_axis_margin", 0.1)
    x_ticks_step = kwargs.get("x_ticks_step", 1)
    y_scale = kwargs.get("y_scale", "linear")

    number_of_subplots = values_all.shape[0]
    number_of_box = values_all.shape[1]

    fig, ax = plt.subplots(nrows=1, ncols=number_of_subplots, sharey=True)
    ax.set_yscale(y_scale)
    if hasattr(ax, "__len__"):
        axes = ax
    else:
        axes = [ax]

    if len(labels) != values_all.shape[1]:
        print("Number of labels does not match number of boxplots to produce!")
        return None
    else:
        for i in range(number_of_subplots):
            values = pd.DataFrame(values_all[i, :, :].reshape(number_of_box, -1)).T
            values.columns = labels

            sns.boxplot(
                data=values,
                whis=[whis_low, whis_high],
                fliersize=2,
                palette="vlag",
                ax=axes[i],
            )

            if axes_plot_titles:
                axes[i].title.set_text(axes_plot_titles[0][i])
                if i == 0:
                    axes[0].set_ylabel(axes_plot_titles[2])
                if isinstance(axes_plot_titles[1], list):
                    axes[i].set_xlabel(axes_plot_titles[1][i])
                else:
                    axes[i].set_xlabel(axes_plot_titles[1])

            # set lower and upper limits based on data.
            if lower_lim is None:
                # take mininmum from data and references if any
                if references_dict is not None:
                    lower_lim = (
                        min(
                            (values.min()).min(),
                            min([ref["lower"] for ref in references_dict.values()]),
                        )
                        - v_axis_margin
                    )
                else:
                    lower_lim = (values.min()).min() - v_axis_margin
            if upper_lim is None:
                # take maximum from data and references if any
                if references_dict is not None:
                    upper_lim = (
                        max(
                            (values.max()).max(),
                            max([ref["upper"] for ref in references_dict.values()]),
                        )
                        + v_axis_margin
                    )
                else:
                    upper_lim = (values.max()).max() + v_axis_margin

                # draw reference lines if any, changing the color for each reference
            if references_dict is not None:
                for i, (ref_name, ref_values) in enumerate(references_dict.items()):
                    axes[i].axhline(
                        ref_values["lower"],
                        color=f"C{i}",
                        linestyle=":",
                        linewidth=1,
                        label=f"{ref_name} 25th percentile",
                    )
                    axes[i].axhline(
                        ref_values["center"],
                        color=f"C{i}",
                        linestyle="--",
                        linewidth=1,
                        label=f"{ref_name} median",
                    )
                    axes[i].axhline(
                        ref_values["upper"],
                        color=f"C{i}",
                        linestyle="-.",
                        linewidth=1,
                        label=f"{ref_name} 75th percentile",
                    )

            axes[i].legend()
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].set_ylim(lower_lim, upper_lim)

            # set scientific notation for y axis
            if lower_lim > 1e2:
                sci_bound = 6 if upper_lim > 1e6 else 3
            else:
                sci_bound = 0
            axes[i].ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(sci_bound, sci_bound),
                useMathText=True,
            )

            # set x axis ticks and labels using log10
            axes[i].set_xticks(list(range(len(labels[::x_ticks_step]))))
            axes[i].set_xticklabels([str(l) for l in labels[::x_ticks_step]])

    plt.gca().spines["left"].set_position(("data", -h_axis_margin))
    plt.gca().spines["bottom"].set_position(("data", lower_lim - v_axis_margin))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_bench_boxplots(
    data_dict,
    metric,
    references_dict=None,
    labels_dict=None,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Create the boxplots for the chosen inversion metrics for all noise scenarios simultaneously. Shows different
    methods results in the same plot.

    data_dict:
        Dictionary containing the data to plot. It should be structured as follows:
        {'rmse': {'small_noise': { 'method_1': np.array([1,2,3]), 'method_2': np.array([4,5,6])},
                 'large_noise': {'method_1': np.array([7,8,9]), 'method_2': np.array([10,11,12])}}}.
    metric:
        the metric to plot. e.g. 'rmse', 'es', 'vs'. One metric at a time.
    references_dict:
        a dictionary containing the reference values for the metrics. It should be structured as follows:
        {'ref1': {'lower': value, 'center': value, 'upper': value},
        'ref2': {'lower': value, 'center': value, 'upper': value}}.
        For relevance of interpretation, the center value should represent a median and lower-upper should represent
        IQR bounds.
    labels_dict:
        a dictionary containing {'plot_title':'', 'x_label':'', 'y_label':''}.
    save_location:
        location where the plot will be stored as pdf.
    dpi:
        resolution of the image in dots per inch (dpi)
    show:
        if True, the plot is shown but not saved. if False, it is only saved
    kwargs:
        configure plotting parameters. For example, 'whis_low', 'whis_high', 'lower_lim', 'upper_lim', 'h_axis_margin',
        'v_axis_margin'.
    """

    import pandas as pd
    import seaborn as sns

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl)

    noise_keys = list(data_dict[metric].keys())
    pred_keys = list(data_dict[metric][noise_keys[0]].keys())

    # make dataframe
    all_values_df = pd.DataFrame(columns=["noise_type", "pred_type", "value"])

    for noise_type in noise_keys:
        for pred_type in pred_keys:
            if noise_type not in data_dict[metric].keys():
                continue
            if pred_type not in data_dict[metric][noise_type].keys():
                continue

            new_data = pd.DataFrame(
                {
                    "noise_type": noise_type,
                    "pred_type": pred_type,
                    "value": data_dict[metric][noise_type][pred_type],
                }
            )
            all_values_df = pd.concat([all_values_df, new_data], ignore_index=True)

        # get kwargs if any or set default values
        whis_low = kwargs.get("whis_low", 2.5)
        whis_high = kwargs.get("whis_high", 97.5)
        lower_lim = kwargs.get("lower_lim", None)
        upper_lim = kwargs.get("upper_lim", None)
        h_axis_margin = kwargs.get("h_axis_margin", 0.7)
        v_axis_margin = kwargs.get("v_axis_margin", 0.1)
        x_ticks_step = kwargs.get("x_ticks_step", 1)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        sns.boxplot(
            x="noise_type",
            y="value",
            data=all_values_df,
            hue="pred_type",
            ax=ax,
            fliersize=2,
            palette="vlag",
            whis=[whis_low, whis_high],
        )

        # set lower and upper limits based on data.
        if lower_lim is None:
            # take mininmum from data and references if any
            if references_dict is not None:
                lower_lim = (
                    min(
                        all_values_df["value"].min(),
                        min([ref["lower"] for ref in references_dict.values()]),
                    )
                    - v_axis_margin
                )
            else:
                lower_lim = all_values_df["value"].min() - v_axis_margin
        if upper_lim is None:
            # take maximum from data and references if any
            if references_dict is not None:
                upper_lim = (
                    max(
                        all_values_df["value"].max(),
                        max([ref["upper"] for ref in references_dict.values()]),
                    )
                    + v_axis_margin
                )
            else:
                upper_lim = all_values_df["value"].max() + v_axis_margin

        # draw reference lines if any, changing the color for each reference
        if references_dict is not None:
            for i, (ref_name, ref_values) in enumerate(references_dict.items()):
                ax.axhline(
                    ref_values["lower"],
                    color=f"C{i}",
                    linestyle=":",
                    linewidth=1,
                    label=f"{ref_name} 25th percentile",
                )
                ax.axhline(
                    ref_values["center"],
                    color=f"C{i}",
                    linestyle="--",
                    linewidth=1,
                    label=f"{ref_name} median",
                )
                ax.axhline(
                    ref_values["upper"],
                    color=f"C{i}",
                    linestyle="-.",
                    linewidth=1,
                    label=f"{ref_name} 75th percentile",
                )

        # set axis labels and title
        if labels_dict is not None:
            ax.set_title(labels_dict.get("plot_title", ""))
            ax.set_xlabel(labels_dict.get("x_label", ""))
            ax.set_ylabel(labels_dict.get("y_label", ""))
        else:
            ax.set_title(f"{metric} boxplots")
            ax.set_xlabel("Noise type")
            ax.set_ylabel(metric)

        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(lower_lim, upper_lim)

        # set scientific notation for y axis
        if lower_lim > 1e2:
            sci_bound = 6 if upper_lim > 1e6 else 3
        else:
            sci_bound = 0
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(sci_bound, sci_bound), useMathText=True
        )

        # set x axis ticks and labels using log10
        ax.set_xticks(list(range(len(noise_keys[::x_ticks_step]))))
        ax.set_xticklabels([label for label in noise_keys[::x_ticks_step]])

        plt.gca().spines["left"].set_position(("data", -h_axis_margin))
        plt.gca().spines["bottom"].set_position(("data", lower_lim - v_axis_margin))

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

        plt.close()


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


def plot_matrices(
    matrices, titles, plot_title=None, save_location=None, dpi=600, show=False, **kwargs
):
    """
    Function to plot along 2 or more matrices.
    :param matrices: a numpy array of the following format [number of matrices, number of rows, number of columns]. All
    matrices must have same number of rows and columns.
    :param titles: list of titles to provide for each matrix plot. Should have length = 'number of matrices'.
    :param save_location: location where to save the generated plot
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    number_of_subplots = matrices.shape[0]
    rows = matrices.shape[1]
    columns = matrices.shape[2]
    fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, sharey=True)

    vmin = np.min(matrices)
    vmax = np.max(matrices)

    for i in range(number_of_subplots):
        m = matrices[i, :, :].reshape(rows, columns)

        im = axes[i].imshow(m)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        im.set_clim(vmin, vmax)
        axes[i].title.set_text(titles[i])
        ax_divider = make_axes_locatable(axes[i])
        cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
        cax.set_xticks([])
        cax.set_yticks([])

        if i == number_of_subplots - 1:
            fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    if plot_title is not None:
        fig.suptitle(plot_title)

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_hv_variograms(
    models,
    args,
    detailed_varios,
    labels=None,
    seperate_vario=False,
    titles=[
        ["Horizontal variograms", "Distance (m)", r"$\gamma$ (Width-direction)"],
        ["Vertical variograms", "Distance (m)", r"$\gamma$ (Depth-direction)"],
    ],
    add_legend=False,
    save_location=None,
    dpi=600,
    show=False,
    **kwargs,
):
    """
    Function to plot the (horizontal and vertical) variograms of the given models.
    :param models: numpy array of the following dimension [number of datasets, samples per dataset, size of model in each dataset]
    :param args: list of arguments needed to format as follows [width, height]
    :param detailed_varios: a list of size 'number of datasets' containing 0 or 1. 1 : indicates that the dataset at the
    specified index needs to be plotted in detail, all variograms need to be plotted. 0: indicates that only mean, min and
    max variograms are to be plotted for the dataset at the specified index.
    :param labels: list of labels to use in the plot's legend
    :param seperate_vario: when True, plot each dataset variograms on a seperate plot. Default: False
    :param titles: array of plot and axes titles as follows [[horiz. vario. plot title, horiz. vario. plot x_axis title,
    horiz. vario. plot y_axis title], [vert. vario. plot title, vert. vario. plot x_axis title,
    vert. vario. plot y_axis title]].
    Default values are provided as follows : [['Horizontal variograms', 'Distance (m)', r'$\gamma$ (y-direction)'],
                                 ['Vertical variograms', 'Distance (m)', r'$\gamma$ (x-direction)']]
    :param save_location: location where to save the generated plots. This function makes two plots, save_location
    should be a list of file names, one for each plot as follows [horizontal variograms file, vertical variograms file]
    :param dpi: resolution of the image in dots per inch (dpi)
    :param kwargs: additional arguments to pass to base_config
    """

    nx = args[0]
    ny = args[1]

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    number_of_datasets = models.shape[0]
    samples_per_dataset = models.shape[1]
    size = models.shape[2]

    vario_h = np.zeros(
        (number_of_datasets, samples_per_dataset, nx)
    )  # array containing horizontal variograms
    vario_v = np.zeros(
        (number_of_datasets, samples_per_dataset, ny)
    )  # array containing vertical variograms

    mean_vario_h = np.zeros((number_of_datasets, nx))
    mean_vario_v = np.zeros((number_of_datasets, ny))

    min_vario_h = np.zeros((number_of_datasets, nx))
    min_vario_v = np.zeros((number_of_datasets, ny))

    max_vario_h = np.zeros((number_of_datasets, nx))
    max_vario_v = np.zeros((number_of_datasets, ny))

    if seperate_vario:
        pass  # todo: implement seperate plots
    else:
        fig_h, ax_h = plt.subplots()
        fig_v, ax_v = plt.subplots()

        # calculate variograms for each dataset
        for i in range(number_of_datasets):
            for j in range(samples_per_dataset):
                m = models[i, j, :].reshape((ny, nx), order="C")
                vario_h[i, j, :] = gs.vario_estimate_structured(m, "y").reshape(
                    (1, -1)
                )  # 'y' here represents the grid-matrix columns direction (width) (read from left to right)
                vario_v[i, j, :] = gs.vario_estimate_structured(m, "x").reshape(
                    (1, -1)
                )  # 'x' here represents the grid-matrix rows direction (height) (read from top to bottom)

                if detailed_varios[i]:
                    # plot detailed vario
                    ax_h.plot(
                        vario_h[i, j, :],
                        linewidth=0.05,
                        color="lightgray",
                        alpha=0.5,
                        label=labels[i],
                    )
                    ax_v.plot(
                        vario_v[i, j, :],
                        linewidth=0.05,
                        color="lightgray",
                        alpha=0.5,
                        label=labels[i],
                    )

            # calculate variogram means, min, max
            mean_vario_h[i, :] = np.mean(vario_h[i, :, :], axis=0)
            mean_vario_v[i, :] = np.mean(vario_v[i, :, :], axis=0)

            min_vario_h[i, :] = np.min(vario_h[i, :, :], axis=0)
            min_vario_v[i, :] = np.min(vario_v[i, :, :], axis=0)

            max_vario_h[i, :] = np.max(vario_h[i, :, :], axis=0)
            max_vario_v[i, :] = np.max(vario_v[i, :, :], axis=0)

            # plot mean, min and max
            ax_h.plot(
                mean_vario_h[i, :],
                linewidth=1.5,
                linestyle=":",
                marker="*",
                label=f"mean_{labels[i]}",
            )
            ax_v.plot(
                mean_vario_v[i, :],
                linewidth=1.5,
                linestyle=":",
                marker="*",
                label=f"mean_{labels[i]}",
            )

            ax_h.plot(
                min_vario_h[i, :],
                linewidth=1.5,
                linestyle="--",
                label=f"min_{labels[i]}",
            )
            ax_v.plot(
                min_vario_v[i, :],
                linewidth=1.5,
                linestyle="--",
                label=f"min_{labels[i]}",
            )

            ax_h.plot(
                max_vario_h[i, :],
                linewidth=1.5,
                linestyle=":",
                marker="+",
                label=f"max_{labels[i]}",
            )
            ax_v.plot(
                max_vario_v[i, :],
                linewidth=1.5,
                linestyle=":",
                marker="+",
                label=f"max_{labels[i]}",
            )

        ylim_max_h = np.max(vario_h) + 0.05
        ylim_max_v = np.max(vario_v) + 0.05

        ax_h.set_title(titles[0][0])
        ax_v.set_title(titles[1][0])

        ax_h.set_xlabel(titles[0][1])
        ax_v.set_xlabel(titles[1][1])

        ax_h.set_ylabel(titles[0][2])
        ax_v.set_ylabel(titles[1][2])

        ax_h.set_xticks([0, 10, 20, 30, 39])
        ax_h.set_xticklabels([0, 1, 2, 3, 4])
        ax_v.set_xticks([0, 10, 20, 30, 40, 49])
        ax_v.set_xticklabels([0, 1, 2, 3, 4, 5])

        ax_h.spines["left"].set_position(("data", 0))
        ax_h.spines["right"].set_visible(False)
        ax_h.spines["top"].set_visible(False)
        if add_legend:
            ax_h.legend()

        ax_v.spines["left"].set_position(("data", 0))
        ax_v.spines["right"].set_visible(False)
        ax_v.spines["top"].set_visible(False)
        if add_legend:
            ax_v.legend()

        ax_h.set_ylim(bottom=0, top=ylim_max_h)
        ax_v.set_ylim(bottom=0, top=ylim_max_v)

        ax_h.set_xlim(left=0, right=nx - 1)
        ax_v.set_xlim(left=0, right=ny - 1)

        plt.tight_layout()
        if show:
            fig_h.show()
            fig_v.show()
        else:
            fig_h.savefig(save_location[0], dpi=dpi, bbox_inches="tight")
            fig_v.savefig(save_location[1], dpi=dpi, bbox_inches="tight")

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


def plot_cov_to_corr(
    cov_matrices, ref_point, args, save_location=None, dpi=600, **kwargs
):
    """
    Function to plot the prior and posterior correlation values from a reference point to all other points,
    based on given prior and posterior covariance matrices.
    :param cov_matrices: a numpy array containing prior and posterior covariance. Format should be [2, size, size]
    :param ref_point: the reference point in the model against which to calculate the correlations with all other
    points in the model. The reference point shoud be provided as number between 0 and size-1. It would be converted
    to the right position in the model upon plotting.
    :param args: a list containing the models shape [height, width]
    :param save_location: folder location where to save the resulting plot
    :param dpi: the image resolution in dots per inch. Default: 600
    :param kwargs: additional arguments to pass to base_config
    """

    model_rows = args[0]
    model_cols = args[1]

    prior_cov = cov_matrices[0, :, :].reshape(
        model_rows * model_cols, model_rows * model_cols
    )
    post_cov = cov_matrices[1, :, :].reshape(
        model_rows * model_cols, model_rows * model_cols
    )

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    ref_row = ref_point // model_cols
    ref_col = ref_point % model_cols

    corrs_prior = np.zeros(model_rows * model_cols)
    corrs_post = np.zeros(model_rows * model_cols)

    for i in range(model_rows * model_cols):
        corrs_prior[i] = prior_cov[i, ref_point] / (
            sqrt(prior_cov[i, i]) * sqrt(prior_cov[ref_point, ref_point])
        )
        corrs_post[i] = post_cov[i, ref_point] / (
            sqrt(post_cov[i, i]) * sqrt(post_cov[ref_point, ref_point])
        )

    vmin = np.min([np.min(corrs_prior), np.min(corrs_post)])
    vmax = np.max([np.max(corrs_prior), np.max(corrs_post)])

    im0 = axes[0].imshow(corrs_prior.reshape(model_rows, model_cols))
    axes[0].scatter(ref_col, ref_row, s=25, c="red", marker="o", alpha=0.2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].title.set_text("Prior correlations")
    im0.set_clim(vmin, vmax)
    ax_divider0 = make_axes_locatable(axes[0])
    cax0 = ax_divider0.append_axes("right", size="5%", pad="2%", frameon=False)
    cax0.set_xticks([])
    cax0.set_yticks([])

    im1 = axes[1].imshow(corrs_post.reshape(model_rows, model_cols))
    axes[1].scatter(ref_col, ref_row, s=25, c="red", marker="o", alpha=0.2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].title.set_text("Posterior correlations")
    im1.set_clim(vmin, vmax)
    ax_divider1 = make_axes_locatable(axes[1])
    cax1 = ax_divider1.append_axes("right", size="5%", pad="2%", frameon=False)
    cax1.set_xticks([])
    cax1.set_yticks([])
    fig.colorbar(im1, cax=cax1, format=tick.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_all_samples(
    ref_x, examples, args, size, vmin, vmax, save_location=None, dpi=600, **kwargs
):
    """
    Function to plot in jpg files the subsurface models images. Useful to make animations.
    To make a gif movie from .jpg images, go to image folder from terminal and run :
    "convert -resize 20% -delay 10 -loop 0 *.jpg myimage.gif"
    (this command requires imagemagick; install with : "sudo apt-get install imagemagick" or "conda install -c conda-forge imagemagick").
    :param examples: a numpy array containing the examples to plot. Expected to be in the format [number of examples, 1, height, width].
    The function expects one channel per image.
    :param args: a list containing the models shape. Expected as [height, width].
    :param size: an integer giving the number of examples to plot.
    :param vmin: a number representing the minimum value in the field observed among all examples. Useful for having a unified colorbar.
    :param vmax: a number representing the maximum value in the field observed among all examples. Useful for having a unified colorbar.
    :param save_location: folder location where to save the resulting plot
    :param dpi: the image resolution in dots per inch. Default: 600
    :param kwargs: additional arguments to pass to base_config
    """

    height = args[0]
    width = args[1]
    set_size = examples.shape[0]

    if examples.shape[1] == height * width:
        examples = examples.reshape(set_size, 1, height, width, order="C")

    for i in range(size):
        mpl, plt, make_axes_locatable, tick = plots_imports()
        base_config(mpl, **kwargs)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        im0 = axes[0].imshow(ref_x)
        im0.set_clim(vmin, vmax)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].title.set_text("Ground truth")
        ax_divider0 = make_axes_locatable(axes[0])
        cax0 = ax_divider0.append_axes("right", size="5%", pad="2%", frameon=False)
        cax0.set_xticks([])
        cax0.set_yticks([])

        im = axes[1].imshow(examples[i, :, :, :].reshape(height, width))
        im.set_clim(vmin, vmax)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        ax_divider = make_axes_locatable(axes[1])
        cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))
        plt.tight_layout()
        plt.savefig(save_location + "/example_{}.jpg".format(i), dpi=dpi)
        plt.close()


def plot_reconstruction(data, dims, title=None, save_location=None, dpi=600, **kwargs):
    """
    Function to plot reconstructed data against the original reference
    :param data: numpy array containing a concatenation of the reference data and its reconstruction.
    :param dims: the dims of the data. It can be either (1, dim) or (dim1, dim2)
    :param save_location: folder location where to save the resulting plot
    :param dpi: the image resolution in dots per inch. Default: 600
    :param kwargs: additional arguments to pass to base_config
    """
    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl, **kwargs)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    height, width = dims

    vmin = np.min(data)
    vmax = np.max(data)

    for i in range(2):
        if i == 0:
            axes[i].title.set_text("Reference")
        else:
            axes[i].title.set_text("Proposed{}".format(" - " + title if title else ""))
        if height == 1:
            im = axes[i].plot(data[i, :].reshape(width))
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["top"].set_visible(False)
            axes[i].set_xlim([0, width - 1])

        else:
            im = axes[i].imshow(data[i, :].reshape(height, width))
            # im.set_clim(vmin, vmax)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            ax_divider = make_axes_locatable(axes[i])
            cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
            cax.set_xticks([])
            cax.set_yticks([])

            if i == 1:
                fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    # read configuration
    parameters_file = (
        "/media/dl-rookie/Data/Final_thesis_results/Data"
        "/exponential_Mu14_Var0p16_CorH25_CorV25_linear_81/parameters.txt "
    )
    config = Config(parameters_file)

    # read data from files
    train_models_file = h5py.File(config.data_folder_location + "/train_models.h5")
    train_models = torch.FloatTensor(train_models_file.get("train_models"))
    train_models_file.close()
    train_truett_file = h5py.File(config.data_folder_location + "/train_truett.h5")
    train_truett = torch.FloatTensor(train_truett_file.get("train_truett"))
    train_truett_file.close()

    idx = 1
    example_x = train_models[idx, :, :, :].numpy().reshape(config.ny, config.nx)
    example_y = train_truett[idx, :].squeeze().numpy()
    save_location_file = config.data_folder_location + "/plot_ex_{}.pdf".format(idx)

    noise_distribution = config.noises_list[1]["distribution"]
    noise_loc = config.noises_list[1]["location"]
    noise_scale = config.noises_list[1]["scale"]

    noisy_tt_folder = (
        config.data_folder_location
        + "/noisy_ttvec_{}_loc{}_scale{}".format(
            noise_distribution, noise_loc, str(noise_scale).replace(".", "p")
        )
    )
    with open(noisy_tt_folder + "/noisy_tt_vec{}".format(idx), "rb") as f:
        noisy_example_y = pickle.load(f)

    plot_example(example_x, example_y, noisy_example_y, save_location_file, dpi=600)
