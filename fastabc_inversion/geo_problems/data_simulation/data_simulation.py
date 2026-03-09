"""
DATA SIMULATION :
-----------------

Script to generate the datasets that are used in the experiments.
The script splits the dataset in train-validation-test.
All configuration parameters are backed up along the data.
The script also generates the noisy travel times (test travel times + noise) vectors that will be used as observation
data for inversion.
"""

import os
import pickle
import random

import fastabc_inversion.geo_problems.data_simulation.prior_setup as ps
import fastabc_inversion.geo_problems.data_simulation.solver_setup as ss
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Set parameters:
# todo: make parameters as external arguments
parameters = {
    "FixedSeed": 42,
    "Dataset size": 10000,  # size of the dataset to generate
    ## configuring the subsurface gaussian field and grid:
    "Gaussian covariance kernel": "matern32",  # covariance kernel choice
    "Number of channels in subsurface images": 1,  # number of channels in subsurface images
    "Width of a grid cell in meters": 0.1,  # width of a cell in meters
    "Number of cells on the horizontal axis (pixels)": 40,  # number of cells on the horizontal axis (number of columns and not the number of rows) (corresponds to pixels)
    "Number of cells on the vertical axis (pixels)": 50,  # number of cells on the vertical axis (number of rows and not the number of columns) (corresponds to pixels)
    "Horizontal correlation length (pixels)": 30,  # horizontal correlation length in number of cells (in pixels)
    "Vertical correlation length (pixels)": 15,  # vertical correlation length in number of cells (in pixels)
    "Gaussian field prior mean": 10,  # prior mean on the field
    "Gaussian field prior variance": 1.96,  # variance at the origin
    "Covariance matrix jitter": 0,
    ## configuring solver
    "solver_type": "eikonal-nl",  # 'linear', #
    "Sources x position": 0,
    "rays": 81,  # number of source x receiver pairs (81: 9S x 9R or 576 : 24S x 24R)
    ## Dataset split: 70 - 20 - 10 Train-Validation-test splits
    "train_split": 0.7,  # 70% training and 30% for validation and test
    "val_split": 0.9,  # validation is 2/3 of the 30% of the whole generated dataset, test is 1/3 of the 30% of the whole generated dataset
    "All noises configurations": [
        {
            "#": "scale is the std",
            "distribution": "Gaussian",
            "location": 0,
            "scale": 0.5,
        },
        {
            "#": "scale is the std",
            "distribution": "Gaussian",
            "location": 0,
            "scale": 2.5,
        },
        {"distribution": "Gumbel", "location": 0, "scale": 2},
    ],
    "rootdir": "/media/dl-rookie/Data/Final_thesis_results",
}

# Read paramters:
cov_kernel = parameters["Gaussian covariance kernel"]
nc = parameters["Number of channels in subsurface images"]
nx = parameters["Number of cells on the horizontal axis (pixels)"]
ny = parameters["Number of cells on the vertical axis (pixels)"]
spacing = parameters["Width of a grid cell in meters"]
lxtrue = parameters["Horizontal correlation length (pixels)"]
lytrue = parameters["Vertical correlation length (pixels)"]
sigma2 = parameters["Gaussian field prior variance"]
mu = parameters["Gaussian field prior mean"]
jitter = parameters["Covariance matrix jitter"]
solver_type = parameters["solver_type"]
sources_x = parameters["Sources x position"]
rays = parameters["rays"]
set_size = parameters["Dataset size"]
train_split = parameters["train_split"]
val_split = parameters["val_split"]
noises_list = parameters["All noises configurations"]
rootdir = parameters["rootdir"]

## setup directories and import necessary files
datadir = rootdir + "/Data"  # location to store the generated data
so_file = rootdir + "/solvers_files/time_2d_new.so"  # so_file for the non-linear solver

# fix seed for all random number generators:
np.random.seed(parameters["FixedSeed"])
os.environ["PYTHONHASHSEED"] = str(parameters["FixedSeed"])
random.seed(parameters["FixedSeed"])
torch.manual_seed(parameters["FixedSeed"])
torch.cuda.manual_seed_all(parameters["FixedSeed"])
np.random.seed(parameters["FixedSeed"])

# generate gaussian random field grid, coordinates differences, prior covariance and prior mean:
prior = ps.prior_setup(nx, ny, lytrue, lytrue, sigma2, cov_kernel, jitter, mu)
CM = prior["covariance_matrix"]  # Gaussian field prior covariance matrix
L = prior["covM_squareRoot"]  # square root of prior covariance matrix
m_prior = prior["prior_mean"]  # Gaussian filed prior mean
ntot = prior["ntot"]

# print('CM shape:', CM.shape)
# print('diagonal elements == {}:{}'.format(sigma2,sum(np.diag(CM) == sigma2)== ntot))

# setup forward solver
if solver_type == "linear":
    if rays == 81:
        start_y = 5.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
        # (.5 to place in middle of the cell, relevant for linear solver)
        step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    if rays == 576:
        start_y = 2.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
        # (.5 to place in middle of the cell, relevant for linear solver)
        step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

    solver_setup_dict = ss.linearSolver_matrix(
        nx, ny, spacing, sources_x, start_y, step_y
    )
    solver_matrix = solver_setup_dict["solver_matrix"]
    ndata = solver_setup_dict["ndata"]

if solver_type == "eikonal-nl":
    if rays == 81:
        start_y = 5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
        # (sets the sources in the upper left corner of the cell)
        step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    if rays == 576:
        start_y = 2  # locate first source/receiver in the vertical direction
        # (sets the sources in the upper left corner of the cell)
        step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

    solver_setup_dict = ss.eikonal_solver_setup(
        nx, ny, sources_x, start_y, step_y, spacing
    )
    ndata = solver_setup_dict["ndata"]

# prepare data structures
truemodels = np.zeros(
    (set_size, nc, ny, nx)
)  # 3D subsurface Gaussian field (set_size) realizations (filled with zeros)
truett = np.zeros(
    (set_size, ndata)
)  # 1D (set_size) first arrival times (filled with zeros)

# data simulation loop
s = 0

with tqdm(total=set_size) as pbar:
    while s < set_size:
        # mtrue = np.random.multivariate_normal(m_prior, CM, 1).T
        mtrue = np.dot(L, np.random.standard_normal(ntot)) + m_prior
        # print(mtrue.shape)
        if not sum(mtrue < 0.0):
            if solver_type == "linear":
                truett[s, :] = solver_matrix @ mtrue.flatten(order="C")
                mtrue = mtrue.reshape(ny, nx, order="C")

            if solver_type == "eikonal-nl":
                mtrue = mtrue.reshape(ny, nx, order="C")
                truett[s, :] = ss.call_eikonal(solver_setup_dict, mtrue, so_file)

            truemodels[s, :, :, :] = mtrue.reshape(1, ny, nx, order="C")
            s = s + 1
            pbar.update(1)


# %%
# Split training, validation, test
train_split_ind = int(np.rint(set_size * train_split))
val_split_ind = int(np.rint(set_size * val_split))

train_models = torch.FloatTensor(truemodels[0:train_split_ind, :, :, :])
train_truett = torch.FloatTensor(truett[0:train_split_ind, :])

val_models = torch.FloatTensor(truemodels[train_split_ind:val_split_ind, :, :, :])
val_truett = torch.FloatTensor(truett[train_split_ind:val_split_ind, :])

test_models = torch.FloatTensor(truemodels[val_split_ind:set_size, :, :, :])
test_truett_noNoise = torch.FloatTensor(truett[val_split_ind:set_size, :])

print("train split _ models:", train_models.shape)
print("train split _ travel times :", train_truett.shape)

print("validation split _ models:", val_models.shape)
print("validation split _ travel times :", val_truett.shape)

print("test split _ models:", test_models.shape)
print("test split _ travel times :", test_truett_noNoise.shape)

# Save data to file
data_folder_location = datadir + "/{}_Mu{}_Var{}_CorH{}_CorV{}_{}_{}".format(
    cov_kernel,
    mu,
    str(round(sigma2, 2)).replace(".", "p"),
    lxtrue,
    lytrue,
    solver_type,
    ndata,
)
try:  # create data folder
    os.mkdir(data_folder_location)
    print("Data folder created...")
except OSError:
    print("Unable to create data folder")

hf = h5py.File(data_folder_location + "/train_models.h5", "w")
hf.create_dataset("train_models", data=train_models)
hf.close()

hf = h5py.File(data_folder_location + "/train_truett.h5", "w")
hf.create_dataset("train_truett", data=train_truett)
hf.close()

hf = h5py.File(data_folder_location + "/val_models.h5", "w")
hf.create_dataset("val_models", data=val_models)
hf.close()

hf = h5py.File(data_folder_location + "/val_truett.h5", "w")
hf.create_dataset("val_truett", data=val_truett)
hf.close()

hf = h5py.File(data_folder_location + "/test_models.h5", "w")
hf.create_dataset("test_models", data=test_models)
hf.close()

hf = h5py.File(data_folder_location + "/test_truett_noNoise.h5", "w")
hf.create_dataset("test_truett_noNoise", data=test_truett_noNoise)
hf.close()

hf = h5py.File(data_folder_location + "/GaussCovMatrix.h5", "w")
hf.create_dataset("GaussianCovarianceMatrix", data=CM)
hf.close()

if solver_type == "linear":
    hf = h5py.File(data_folder_location + "/linearForwardMatrix.h5", "w")
    hf.create_dataset("LinearSolverMatrix", data=solver_matrix)
    hf.close()

# Backup parameters dictionary to file:
with open(data_folder_location + "/parameters.txt", "w") as data:
    data.write(str(parameters))

# Prepare noisy test travel times:

test_set_size = test_truett_noNoise.shape[0]

for noise_i in noises_list:
    ## Read noise configuration
    noise_distribution = noise_i["distribution"]
    noise_loc = noise_i["location"]
    noise_scale = noise_i["scale"]

    noisy_tt_folder = data_folder_location + "/noisy_ttvec_{}_loc{}_scale{}".format(
        noise_distribution, noise_loc, str(noise_scale).replace(".", "p")
    )
    try:  # create noise test travel times folder
        os.mkdir(noisy_tt_folder)
        print("Noisy data folder created for {} noise...".format(noise_distribution))
    except OSError:
        print(
            "Unable to create Noisy data folder for {} noise...".format(
                noise_distribution
            )
        )

    for idx in range(test_set_size):
        if noise_distribution == "Gaussian":
            noise = noise_scale * np.random.randn(ndata) + noise_loc
        elif noise_distribution == "Gumbel":
            noise = np.random.gumbel(loc=noise_loc, scale=noise_scale, size=ndata)
        else:
            print("Unknown distribution!")
            break
        noisy_tt = test_truett_noNoise[idx,] + noise

        # Save noisy travel times to file
        with open(noisy_tt_folder + "/noisy_tt_vec{}".format(idx), "wb") as f:
            pickle.dump(noisy_tt, f)

        with open(noisy_tt_folder + "/noise_configuration", "w") as f:
            f.write(str(noise_i))
