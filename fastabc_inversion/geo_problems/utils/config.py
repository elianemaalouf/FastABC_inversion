"""
UTILES:
------

Set of tools/methods often called for setting up things (e.g., read parameters, set random seed. etc.)
"""
import ast
import os
import random

import fastabc_inversion.geo_problems.data_simulation.prior_setup as ps
import fastabc_inversion.geo_problems.data_simulation.solver_setup as ss
import numpy as np
import torch


class Config:
    """Class to read environment configuration parameters for Goephysics applications."""

    def __init__(self, parameters_file, setup_solver=True, setup_prior=True):
        """
        Initializes Config instance attributes from values stored in a configuration file on disk.
        :param parameters_file: configuration file
        """
        with open(parameters_file) as f:
            self.parameters = f.read()

        self.parameters = ast.literal_eval(self.parameters)

        # Read paramters:
        self.cov_kernel = self.parameters["Gaussian covariance kernel"]
        self.nc = self.parameters["Number of channels in subsurface images"]
        self.nx = self.parameters["Number of cells on the horizontal axis (pixels)"]
        self.ny = self.parameters["Number of cells on the vertical axis (pixels)"]
        self.spacing = self.parameters["Width of a grid cell in meters"]
        self.lxtrue = self.parameters["Horizontal correlation length (pixels)"]
        self.lytrue = self.parameters["Vertical correlation length (pixels)"]
        self.sigma2 = self.parameters["Gaussian field prior variance"]
        self.mu = self.parameters["Gaussian field prior mean"]
        self.jitter = self.parameters["Covariance matrix jitter"]
        self.solver_type = self.parameters["solver_type"]
        self.sources_x = self.parameters["Sources x position"]
        self.rays = self.parameters["rays"]
        self.set_size = self.parameters["Dataset size"]
        self.train_split = self.parameters["train_split"]
        self.val_split = self.parameters["val_split"]
        self.noises_list = self.parameters["All noises configurations"]
        self.rootdir = self.parameters["rootdir"]

        # fix seed for all random number generators:
        np.random.seed(self.parameters["FixedSeed"])
        os.environ["PYTHONHASHSEED"] = str(self.parameters["FixedSeed"])
        random.seed(self.parameters["FixedSeed"])
        torch.manual_seed(self.parameters["FixedSeed"])
        torch.cuda.manual_seed_all(self.parameters["FixedSeed"])
        np.random.seed(self.parameters["FixedSeed"])
        print("Random seed fixed to:{}".format(self.parameters["FixedSeed"]))

        if setup_prior:
            # set up prior mean and covariance
            prior = ps.prior_setup(
                self.nx,
                self.ny,
                self.lytrue,
                self.lytrue,
                self.sigma2,
                self.cov_kernel,
                self.jitter,
                self.mu,
            )
            self.CM = prior[
                "covariance_matrix"
            ]  # Gaussian field prior covariance matrix
            self.L = prior["covM_squareRoot"]  # square root of prior covariance matrix
            self.m_prior = prior["prior_mean"]  # Gaussian filed prior mean
            self.ntot = prior["ntot"]
        else:
            self.ntot = self.nx * self.ny

        if setup_solver:
            # setup forward solver _ only linear solver
            if self.solver_type == "linear":
                if self.rays == 81:
                    start_y = 5.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                    # (.5 to place in middle of the cell, relevant for linear solver)
                    step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                if self.rays == 576:
                    start_y = 2.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                    # (.5 to place in middle of the cell, relevant for linear solver)
                    step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

                self.solver_setup_dict = ss.linearSolver_matrix(
                    self.nx, self.ny, self.spacing, self.sources_x, start_y, step_y
                )
                self.solver_matrix = self.solver_setup_dict["solver_matrix"]
                self.ndata = self.solver_setup_dict["ndata"]

            if self.solver_type == "eikonal-nl":
                if self.rays == 81:
                    start_y = 5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                    # (sets the sources in the upper left corner of the cell)
                    step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                if self.rays == 576:
                    start_y = (
                        2  # locate first source/receiver in the vertical direction
                    )
                    # (sets the sources in the upper left corner of the cell)
                    step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

                self.solver_setup_dict = ss.eikonal_solver_setup(
                    self.nx, self.ny, self.sources_x, start_y, step_y, self.spacing
                )
                self.ndata = self.solver_setup_dict["ndata"]
                self.so_file = (
                    self.rootdir + "/solvers_files/time_2d_new.so"
                )  # so_file for the non-linear solver
        else:
            self.ndata = self.rays

        ## setup directories and import necessary files
        self.datadir = self.rootdir + "/Data"
        self.data_folder_location = (
            self.datadir
            + "/{}_Mu{}_Var{}_CorH{}_CorV{}_{}_{}".format(
                self.cov_kernel,
                self.mu,
                str(round(self.sigma2, 2)).replace(".", "p"),
                self.lxtrue,
                self.lytrue,
                self.solver_type,
                self.ndata,
            )
        )


class BenchConfig:
    """Class to read environment configuration parameters for benchmark examples."""

    def __init__(self, parameters_file):
        """
        Initializes Config instance attributes from values stored in a configuration file on disk.
        :param parameters_file: configuration file
        """
        with open(parameters_file) as f:
            self.parameters = f.read()

        self.parameters = ast.literal_eval(self.parameters)

        # fix seed for all random number generators:
        np.random.seed(self.parameters["FixedSeed"])
        os.environ["PYTHONHASHSEED"] = str(self.parameters["FixedSeed"])
        random.seed(self.parameters["FixedSeed"])
        torch.manual_seed(self.parameters["FixedSeed"])
        torch.cuda.manual_seed_all(self.parameters["FixedSeed"])
        np.random.seed(self.parameters["FixedSeed"])
        print("Random seed fixed to:{}".format(self.parameters["FixedSeed"]))


if __name__ == "__main__":
    parameters_file = "/media/dl-rookie/Data/Final_thesis_results/Data/exponential_Mu14_Var0p16_CorH25_CorV25_linear_81/parameters.txt"
    params = Config(parameters_file)
