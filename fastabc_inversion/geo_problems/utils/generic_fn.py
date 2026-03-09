"""
File containing generic utility functions
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
"""

import h5py
import numpy as np
import torch


def import_dataset(datasets_list, config_obj, train_set_size, subset_train=None):
    """
    Import datasets from the specified list.
    :param datasets_list: list of dataset names, should contain "train" at least
    :param config_obj: config object containing the data folder location
    :param train_set_size: size of the training set
    :param subset_train: number of training samples to select randomly from the training set
    :return : list of imported datasets
    """
    if "train" not in datasets_list:
        raise ValueError("The dataset list should contain 'train' at least")

    result = {}

    if "train" in datasets_list:
        idx_train = np.random.choice(
            np.arange(train_set_size), size=subset_train, replace=False
        )
        train_models_file = h5py.File(
            config_obj.data_folder_location + "/train_models.h5"
        )
        train_models_all = (
            torch.tensor(train_models_file.get("train_models"), dtype=torch.float64)
            .numpy()
            .reshape(-1, config_obj.nx * config_obj.ny)
        )
        train_models = train_models_all[idx_train, :]
        train_models_file.close()
        train_truett_file = h5py.File(
            config_obj.data_folder_location + "/train_truett.h5"
        )
        train_truett_all = (
            torch.tensor(train_truett_file.get("train_truett"), dtype=torch.float64)
            .numpy()
            .reshape(-1, config_obj.rays)
        )
        train_truett_noiseless = train_truett_all[idx_train, :]
        train_truett_file.close()

        x_mean = train_models.mean(axis=0)
        y_mean = train_truett_noiseless.mean(axis=0)

        result["means"] = [x_mean, y_mean]
        result["train"] = [train_models, train_truett_noiseless]

    if "val" in datasets_list:
        # import validation sets
        val_models_file = h5py.File(config_obj.data_folder_location + "/val_models.h5")
        val_models = torch.tensor(
            val_models_file.get("val_models"), dtype=torch.float64
        ).numpy()
        val_models = val_models.reshape(-1, config_obj.nx * config_obj.ny)
        val_models_file.close()
        val_truett_file = h5py.File(config_obj.data_folder_location + "/val_truett.h5")
        val_truett_noiseless = torch.tensor(
            val_truett_file.get("val_truett"), dtype=torch.float64
        ).numpy()
        val_truett_noiseless = val_truett_noiseless.reshape(-1, config_obj.rays)
        val_truett_file.close()

        result["val"] = [val_models, val_truett_noiseless]

    if "test" in datasets_list:
        test_models_file = h5py.File(
            config_obj.data_folder_location + "/test_models.h5"
        )
        test_models = torch.tensor(
            test_models_file.get("test_models"), dtype=torch.float64
        ).numpy()
        test_models = test_models.reshape(-1, config_obj.nx * config_obj.ny)
        test_models_file.close()
        test_truett_file = h5py.File(
            config_obj.data_folder_location + "/test_truett_noNoise.h5"
        )
        test_truett_noiseless = torch.tensor(
            test_truett_file.get("test_truett_noNoise"), dtype=torch.float64
        ).numpy()
        test_truett_noiseless = test_truett_noiseless.reshape(-1, config_obj.rays)
        test_truett_file.close()

        result["test"] = [test_models, test_truett_noiseless]

    return result


def get_noise(config_obj, noise_label):
    """
    Get the noise parameters from the config object
    :param config_obj: config object containing the noise parameters
    :param noise_label: label of the noise
    :return: noise parameters
    """
    if noise_label == "small_gauss":
        return config_obj.noises_list[0]
    elif noise_label == "large_gauss":
        return config_obj.noises_list[1]
    elif noise_label == "gumbel":
        return config_obj.noises_list[2]
    else:
        raise ValueError("Unknown noise label: {}".format(noise_label))


def save_to_disk(data, file_path, _pickle=True, _text=False):
    # pickle the data
    if _pickle:
        import pickle

        # add .pkl extension if not present
        if not file_path.endswith(".pkl"):
            file_path += ".pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    if _text:
        # add .txt extension if not present
        if not file_path.endswith(".txt"):
            file_path += ".txt"

        with open(file_path, "w") as f:
            if not isinstance(data, str):
                data = str(data)
            f.write(data)


def load_from_disk(file_path):
    import pickle

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data
