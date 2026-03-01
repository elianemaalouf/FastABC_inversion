"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Utility functions.
"""
import torch
from torch import nn


def params_init(m, args=("relu", None), verbose=False):
    """
    Function to randomly initialize the weights in the NN based on Xavier initialization.
    BatchNorm layers are initialized with default values via the reset_parameters() function.
    :param m: the layer for which to initialize the weights
    :param args: a tuple of 2 elements to configure the initialization function.
    The tuple contains:
        - first element: type of gain to be used : 'ReLU' 'Leaky Relu' etc.
        If not None, this is the first parameter of the second parameter of torch.nn.init.calculate_gain(nonlinearity, param=None)
        if None, then the default is maintained.
        - second element: the value to use for the bias. Leave it to None if the value should be 0 by default.

    Check for type of gain : https://pytorch.org/docs/2.1/nn.init.html#torch-nn-init
    :param verbose: whether to print to console the values of the weights, before and after initialization (for debugging)
    """

    fn_params = args
    _name_map = {
        "relu": "relu",
        "leaky_relu": "leaky_relu",
        "prelu": "leaky_relu",
        "constrained_prelu": "leaky_relu",
        "swish": "swish",
        "silu": "swish",
        "beta_swish": "swish",
        "constrained_beta_swish": "swish",
    }
    name_map = _name_map[fn_params[0]]
    if name_map in ["relu", "leaky_relu"]:
        param = 0.25 if _name_map[fn_params[0]] == "leaky_relu" else None
        gain = torch.nn.init.calculate_gain(name_map, param=param)
    elif name_map in ["swish"]:
        gain = 1.75
    else:
        raise NotImplementedError("Unknown activation function {}".format(fn_params[0]))

    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        if verbose:  # before initialization
            print("Before init")
            print("m.weight:", m.weight)
            print("m.bias:", m.bias)

        if fn_params[0] is None:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            if fn_params[1] is not None:
                torch.nn.init.constant_(m.bias, fn_params[1])
            else:
                torch.nn.init.zeros_(m.bias)

        if verbose:  # after initialization
            print("After init")
            print("m.weight:", m.weight)
            print("m.bias:", m.bias)

    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.reset_parameters()


def toggle_spectral_norm(model, use_spectral_norm, layer_types=None, verbose=False):
    """
    Toggles spectral normalization on specific layers of the given model. This function
    iterates through the modules of the `model` and applies or ensures that spectral
    normalization is added to eligible layers (`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`,
    or `nn.Linear`). If `use_spectral_norm` is set to `True`, spectral normalization
    is applied where it is not already present. Otherwise, no operation is performed.

    :param model: The PyTorch model containing the modules to inspect and modify.
    :param use_spectral_norm: A flag indicating whether to apply spectral normalization.
    """

    from torch.nn.utils import spectral_norm

    if layer_types is None:
        layer_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

    for module in model.modules():
        # Check if the module is a layer where spectral norm can be applied
        if isinstance(module, layer_types):
            if use_spectral_norm:
                # Apply spectral normalization if not already applied
                if not hasattr(module, "weight_orig"):
                    if verbose:
                        print(f"Applying spectral norm to layer: {module}")
                    spectral_norm(module)


def print_spectral_norm_status(model, layer_types=None):
    """
    Prints the spectral normalization status of all layers in the given model that are instances of specific types
    of PyTorch modules, such as convolutional or linear layers.

    :param model: The PyTorch model instance whose modules are to be inspected for spectral normalization.
    """
    if layer_types is None:
        layer_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            status = (
                "Spectral Norm applied"
                if hasattr(module, "weight_orig")
                else "No Spectral Norm"
            )
            print(f"{name}: {status}")


def load_experiment_from_file(experiment_checkpoint_file):
    """
    Load the experiment from file.
    :param experiment_checkpoint_file: path to the experiment checkpoint file
    :return: the experiment object
    """
    import pickle

    with open(experiment_checkpoint_file, "rb") as f:
        experiment = pickle.load(f)
    return experiment


def norm_fn_selector(type="lpp"):
    """
    Select the function based on the type
    :param type: string indicating which function to use. Options are 'mse', 'rmse', 'lpp' (=lp^p), 'lp'.
    Default is 'lpp'.
    :return: the function to use
    """

    import fastabc_inversion.conditional_generation.utils.torch_distances as td

    _map = {
        "mse": td.mse_torch,
        "rmse": td.rmse_torch,
        "lpp": td.lpp_torch,  # lp^p
        "lp": td.lp_torch,
        "ce": td.cross_entropy_torch,
    }

    return _map[type]


def compute_stats(np_array):
    """
    Compute statistics for an array.
    Statistics computed are mean, median, 25th and 75th percentiles, 2.5th and 97.5th percentiles.
    :param np_array: 1D numpy array
    NOTE: same function in the file: ThesisCodes/Geo_Problems/Diagnostics.py
    """
    import numpy as np

    if np_array is not None:
        mean = np.mean(np_array)
        std = np.std(np_array)
        median = np.median(np_array)
        q25 = np.quantile(np_array, q=0.25, interpolation="nearest")
        q75 = np.quantile(np_array, q=0.75, interpolation="nearest")
        q025 = np.quantile(np_array, q=0.025, interpolation="nearest")
        q975 = np.quantile(np_array, q=0.975, interpolation="nearest")

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "q25": q25,
            "q75": q75,
            "q025": q025,
            "q975": q975,
        }
    else:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "q025": None,
            "q975": None,
        }
