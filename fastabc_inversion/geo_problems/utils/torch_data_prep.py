"""
Module containing functions that are useful to prepare or manipulate the data when in torch tensors.
Written by Eliane Maalouf
"""
import torch

_AVAILABLE_NORMALIZATION_FUNCTIONS = ["min_max", "standardize"]


def min_max_normalize(
    data, lower_p=None, upper_p=None, lower_value=None, upper_value=None
):
    """
    Function that performs min-max normalization on data. If lower_p and upper_p are provided, this function uses quantiles
    to compute the normalization. Otherwise, it uses the minimum and maximum values of the data.
    The function expects data in shape (# of observation, dim of data).
    Each dimension is normalized separately.
    :param data: data tensor to be normalized
    :param lower_p: lower percentile of data to use for robust normalization. Should be provided as percentage,
                    e.g, 1% or 5%. If None, minimum is used. A quantile is computed for each dimension.
    :param upper_p: upper percentile of data to use for robust normalization. Should be provided as percentage,
                    e.g, 99% or 95%. If None, maximum is used. A quantile is computed for each dimension.
    :param lower_value: lower value to use for normalization. Used for example when normalizing validation data.
           Shape should be (1, dim of data) for broadcasting to work correctly.
    :param upper_value: upper value to use for normalization. Used for example when normalizing validation data.
           Shape should be (1, dim of data) for broadcasting to work correctly.
    :return: normalized data
    """
    dim = 0
    if lower_value is not None:
        lower_value = lower_value
    else:
        if lower_p is None:
            lower_value = torch.min(data, dim=dim)[0]
        else:
            lower_value = torch.quantile(
                data, lower_p / 100, dim=dim, interpolation="nearest"
            )

    lower_value = lower_value.reshape(1, -1)

    if upper_value is not None:
        upper_value = upper_value
    else:
        if upper_p is None:
            upper_value = torch.max(data, dim=dim)[0]
        else:
            upper_value = torch.quantile(data, upper_p / 100, dim=dim)

    upper_value = upper_value.reshape(1, -1)

    denominator = upper_value - lower_value
    # Avoid division by zero
    denominator = torch.where(
        denominator == 0, torch.ones_like(denominator), denominator
    )

    normalized_data = (data - lower_value) / denominator

    return normalized_data, lower_value, upper_value


def min_max_unnormalize(data, lower_value, upper_value):
    """
    Function that un-normalizes data that was normalized using min-max normalization.
    The function expects data in shape (# of observation, dim of data).
    Each dimension is un-normalized separately.
    :param data: data tensor to be un-normalized
    :param lower_value: lower value used for normalization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :param upper_value: upper value used for normalization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :return: un-normalized data
    """
    if data.shape[1] != lower_value.shape[1] or data.shape[1] != upper_value.shape[1]:
        raise ValueError("Data, lower_value, and upper_value must have the same shape.")

    device = data.device
    lower_value = lower_value.to(device)
    upper_value = upper_value.to(device)
    denominator = upper_value - lower_value
    # Avoid division by zero
    denominator = torch.where(
        denominator == 0, torch.ones_like(denominator), denominator
    )

    unnormalized_data = data * denominator + lower_value

    return unnormalized_data


def standardize(data, mean, std):
    """
    Function that standardizes data using the mean and standard deviation provided.
    The function expects data in shape (# of observation, dim of data).
    Each dimension is standardized separately.
    :param data: data tensor to be standardized
    :param mean: mean of the data to use for standardization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :param std: standard deviation of the data to use for standardization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :return: standardized data
    """
    if data.shape[1] != mean.shape[1] or data.shape[1] != std.shape[1]:
        raise ValueError("Data, mean, and std must have the same shape.")
    std = torch.where(std == 0, torch.ones_like(std), std)

    return (data - mean) / std


def un_standardize(data, mean, std):
    """
    Function that un-standardizes data using the mean and standard deviation provided.
    The function expects data in shape (# of observation, dim of data).
    Each dimension is un-standardized separately.
    :param data: data tensor to be un-standardized
    :param mean: mean of the data to use for un-standardization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :param std: standard deviation of the data to use for un-standardization. Shape should be (1, dim of data) for broadcasting to work correctly.
    :return: un-standardized data
    """
    if data.shape[1] != mean.shape[1] or data.shape[1] != std.shape[1]:
        raise ValueError("Data, mean, and std must have the same shape.")
    std = torch.where(std == 0, torch.ones_like(std), std)
    return data * std + mean


def normalize(data, normalize_dict):
    """
    Function that normalizes data according to the normalize_dict provided.
    :param data: data tensor to be standardized. Shape should be (# of observation, dim of data).
    :param normalize_dict: dictionary containing the normalization information. It should have the following keys:
            "function": "min_max" or "standardize"
            "kwargs": dictionary containing the arguments for the normalization function.
            For min_max, provide lower_p and upper_p as percentage values or lower_value and upper_value as tensors.
            For standardize, provide mean and std tensors.
    :return: normalized data
    """
    if normalize_dict["function"] not in _AVAILABLE_NORMALIZATION_FUNCTIONS:
        raise ValueError(
            "Normalization function not available. Choose from: ",
            _AVAILABLE_NORMALIZATION_FUNCTIONS,
        )

    if normalize_dict["function"] == "min_max":
        return min_max_normalize(data, **normalize_dict["kwargs"])[0]

    if normalize_dict["function"] == "standardize":
        return standardize(data, **normalize_dict["kwargs"])


def un_normalize(data, normalize_dict):
    """
    Function that un-normalizes data according to the normalize_dict provided.
    :param data: data tensor to be standardized. Shape should be (# of observation, dim of data).
    :param dictionary containing the normalization information. It should have the following keys:
            "function": "min_max" or "standardize"
            "kwargs": dictionary containing the arguments for the normalization function.
            For min_max, provide lower_value and upper_value as tensors.
            For standardize, provide mean and std tensors.
    :return: unnormalized data
    """
    if normalize_dict["function"] not in _AVAILABLE_NORMALIZATION_FUNCTIONS:
        raise ValueError(
            "Normalization function not available. Choose from: ",
            _AVAILABLE_NORMALIZATION_FUNCTIONS,
        )

    if normalize_dict["function"] == "min_max":
        return min_max_unnormalize(data, **normalize_dict["kwargs"])

    if normalize_dict["function"] == "standardize":
        return un_standardize(data, **normalize_dict["kwargs"])


if __name__ == "__main__":
    # test min_max_normalizer
    data = torch.tensor([[1, 2, 3], [1, 5, 6], [1, 8, 100]], dtype=torch.float)
    lower_p = 1
    upper_p = 99
    normalized_data, lower_value, upper_value = min_max_normalize(
        data, lower_p, upper_p
    )
    print("Normalized data: ", normalized_data)

    # test min_max_unnormalizer
    unnormalized_data = min_max_unnormalize(normalized_data, lower_value, upper_value)
    print("Unnormalized data: ", unnormalized_data)

    # test standardizer
    mean = torch.mean(data, dim=0).unsqueeze(0)
    std = torch.std(data, dim=0).unsqueeze(0)
    standardized_data = standardize(data, mean, std)
    print("Standardized data: ", standardized_data)

    # test un_standardizer
    unstandardized_data = un_standardize(standardized_data, mean, std)
    print("Unstandardized data: ", unstandardized_data)
