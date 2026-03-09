"""
sover_setup
-----------

Module to setup solvers and calls to those solvers
"""
import numpy as np
from fastabc_inversion.geo_problems.solvers.eikonal_solver import time2d_py
from fastabc_inversion.geo_problems.solvers.tomokernel_straight import \
    tomokernel_straight_2D


def linearSolver_matrix(nx, ny, spacing, sources_x, start_y, step_y):
    """linearSolver_matrix
    Function to generate the linear solver matrix given the subsurface domain grid

    :param nx: number of cells on the horizontal axis (number of columns and not the number of rows) (corresponds to pixels)
    :param ny: number of cells on the vertical axis (number of rows and not the number of columns) (corresponds to pixels)
    :param spacing: width of a cell in meters
    :param sources_x: position of the sources on the horizontal axis
    :param start_y: locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :param step_y: distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :return: dictionnary containing the following keys {'solver_matrix' (the solver matrix) ,
    'ndata' (length of data/measurement vector)}
    """
    x = np.arange(0, (nx * spacing) + spacing, spacing)  # in meters
    y = np.arange(0, (ny * spacing) + spacing, spacing)

    sourcex = sources_x
    sourcey = (
        np.arange(start_y, ny, step_y) * spacing
    )  # sets the sources in the middle of each cell, in meters
    receiverx = nx * spacing
    receivery = (
        np.arange(start_y, ny, step_y) * spacing
    )  # sets the receivers in the middle of each cell, in meters
    nsource = len(sourcey)
    nreceiver = len(receivery)

    ndata = nsource * nreceiver  # number of rays

    data = np.zeros((ndata, 4))
    # Calculate acquisition geometry (multiple-offset gather)
    for jj in range(0, nsource):
        for ii in range(0, nreceiver):
            data[jj * nreceiver + ii, :] = np.array(
                [sourcex, sourcey[jj], receiverx, receivery[ii]]
            )
        # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
    A = tomokernel_straight_2D(
        data, x, y
    )  # Distance of ray-segment in each cell for each ray
    A = np.array(A.todense())
    del data

    return {"solver_matrix": A, "ndata": ndata}


def eikonal_solver_setup(nx, ny, sources_x, start_y, step_y, spacing):
    """eikonal_solver_setup
    Function to setup eikonal solver acquisition geometry

    :param nx: number of cells on the horizontal axis (number of columns and not the number of rows) (corresponds to pixels)
    :param ny: number of cells on the vertical axis (number of rows and not the number of columns) (corresponds to pixels)
    :param sources_x: position of the sources on the horizontal axis
    :param start_y: locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :param step_y: distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :param spacing: width of a cell in meters
    :return: configuration dictionnary containing the following keys {'nx' (width), 'ny' (height), 'spacing' (unit to convert from pixels to meters),
    'number of sources', 'number of receivers', 'soucrces horizontal coordinates', 'sources vertical coordinates',
    'receivers horizontal coordinates', 'receivers vertical coordinates',
    'ndata' (length of data/measurement vector)}
    """

    sourcex = sources_x
    sourcey = np.arange(
        start_y, ny, step_y
    )  # sets the sources in the upper left corner of the cell
    receiverx = nx
    receivery = np.arange(
        start_y, ny, step_y
    )  # sets the receivers in the upper left corner of the cell
    nsource = len(sourcey)
    nreceiver = len(receivery)

    ndata = nsource * nreceiver

    return {
        "nx": nx,
        "ny": ny,
        "spacing": spacing,
        "number of sources": nsource,
        "number of receivers": nreceiver,
        "soucrces horizontal coordinates": sourcex,
        "sources vertical coordinates": sourcey,
        "receivers horizontal coordinates": receiverx,
        "receivers vertical coordinates": receivery,
        "ndata": ndata,
    }


def prep_solver(solver_type, args):
    """
    Function to setup the solver dictionnary that contains solver's necessary configurations.
    :param solver_type: choose a string 'linear' or 'eikonal-nl'.
    :param args: a list of parameters that need to be provided as follows [number of rays, width, height, spacing, horizontal_locatio_sources]
    :return: a dictionnary containing as formated by the function linearSolver_matrix() when solver_type = 'linear'
    and as formated by the funciton eikonal_solver_setup() when solver_type = 'eikonal-nl'
    """

    rays = args[0]
    nx = args[1]
    ny = args[2]
    spacing = args[3]
    sources_x = args[4]

    if solver_type == "linear":
        if rays == 81:
            start_y = 5.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
            # (.5 to place in middle of the cell, relevant for linear solver)
            step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
        if rays == 576:
            start_y = 2.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
            # (.5 to place in middle of the cell, relevant for linear solver)
            step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

        solver_setup_dict = linearSolver_matrix(
            nx, ny, spacing, sources_x, start_y, step_y
        )

    if solver_type == "eikonal-nl":
        if rays == 81:
            start_y = 5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
            # (sets the sources in the upper left corner of the cell)
            step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
        if rays == 576:
            start_y = 2  # locate first source/receiver in the vertical direction
            # (sets the sources in the upper left corner of the cell)
            step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

        solver_setup_dict = eikonal_solver_setup(
            nx, ny, sources_x, start_y, step_y, spacing
        )

    return solver_setup_dict


def call_eikonal(solver_setup_dict, model, so_file):
    """call_eikonal
    Function to call the eikonal solver on a given subsurface model

    :param solver_setup_dict: solver setup dictionary
    :param model: subsurface model
    :param so_file:
    :return: first arrival times relative to the provided model
    """
    nx = solver_setup_dict["nx"]
    ny = solver_setup_dict["ny"]
    spacing = solver_setup_dict["spacing"]
    nsource = solver_setup_dict["number of sources"]
    nreceiver = solver_setup_dict["number of receivers"]
    sourcex = solver_setup_dict["soucrces horizontal coordinates"]
    sourcey = solver_setup_dict["sources vertical coordinates"]
    receiverx = solver_setup_dict["receivers horizontal coordinates"]
    receivery = solver_setup_dict["receivers vertical coordinates"]
    ndata = solver_setup_dict["ndata"]

    first_arrivals, _, err = time2d_py(
        nx,
        ny,
        nsource,
        nreceiver,
        xs=sourcex,
        ys=sourcey,
        rz=receivery,
        rx=receiverx,
        s_model=model,
        spacing=spacing,
        so_file=so_file,
    )
    if err:
        return None  # todo: raise exception instead
    else:
        return first_arrivals
