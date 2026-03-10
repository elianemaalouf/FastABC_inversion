## Smoothing and Derivatives for Plot Points

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import (Akima1DInterpolator, CubicSpline,
                               PchipInterpolator, interp1d, make_interp_spline)

_ALLOWED_INTERPOLANTS = {
    "cubic_spline": CubicSpline,
    "b_spline": make_interp_spline,
    "monotone_pchip": PchipInterpolator,
    "monotone_akima": Akima1DInterpolator,
}


def check_sorted(x):
    # Check for NaN or infinite values
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("`sorted_b_line` contains NaN or infinite values.")

    # Ensure sorted_b_line is strictly increasing by adding a small epsilon if necessary
    epsilon = 1e-10
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + epsilon

    return x


def numerical_derivative(f, x, h=1e-5):
    """
    Compute the numerical derivative of a function at a given point.
    :param f: function to differentiate
    :param x: point at which to compute the derivative
    :param h: precision of the derivative
    :return:
    """
    x = np.clip(x, a_min=x.min() + h, a_max=x.max() - h)
    return (f(x + h) - f(x - h)) / (2 * h)


def compute_curvature(first_derivative, second_derivative):
    """
    Compute the curvature of a function given its first and second derivatives.
    :param first_derivative: first derivative of the function
    :param second_derivative: second derivative of the function
    :return: curvature of the function
    """
    return np.abs(second_derivative) / (1 + first_derivative**2) ** 1.5


def approx_curve(
    x,
    y,
    interpolant="cubic_spline",
    new_points=1000,
    logspace=False,
    precision=1e-05,
    **kwargs,
):
    """
    Generic function to interpolate a curve using different interpolation techniques. Computes the first and second
    derivatives, as well as the point of highest curvature. !! Interpolation does not smooth the data, it just fits a
    curve through the points. Use curve_smoother() for smoothing and to compute stabler derivatives.
    :param x: x values
    :param y: y values
    :param interpolant: type of interpolant to use (cubic_splie, b_spline, monotone_pchip, monotone_akima)
    :param new_points: number of new points to generate for the spline
    :param precision: precision of the numerical derivative
    :param kwargs: extra arguments for the interpolation function
    :return:
    """

    x = check_sorted(x)

    if interpolant.lower() not in _ALLOWED_INTERPOLANTS.keys():
        raise ValueError(f"Interpolant must be one of {_ALLOWED_INTERPOLANTS}")

    interpolator = _ALLOWED_INTERPOLANTS[interpolant.lower()](x, y, **kwargs)

    # Generate new x values for a smooth curve
    if logspace:
        x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), num=new_points)
    else:
        x_new = np.linspace(x.min(), x.max(), num=new_points)

    y_new = interpolator(x_new)

    # Calculate the first and second derivatives
    first_derivative = numerical_derivative(interpolator, x_new, h=precision)

    # smooth the first derivative
    # deriv_interpolator = _ALLOWED_INTERPOLANTS[interpolant.lower()](x_new, first_derivative, **kwargs)
    deriv_interpolator = interp1d(x_new, first_derivative, kind="linear")
    first_derivative_smoothed = deriv_interpolator(x_new)

    second_derivative = numerical_derivative(deriv_interpolator, x_new, h=precision)

    curvature = compute_curvature(first_derivative_smoothed, second_derivative)

    max_curvature_index = np.argmax(curvature)  # on the new x

    # retrieve in original x the point closest to the max curvature
    x_max_curvature = x_new[max_curvature_index]
    original_closest_index = np.argmin(np.abs(x - x_max_curvature))

    return {
        "x_new": x_new,
        "y_new": y_new,
        "first_derivative": first_derivative_smoothed,
        "second_derivative": second_derivative,
        "curv_y": curvature,
        "id_max": max_curvature_index,  # on the new x
        "id_max_origin": original_closest_index,
        "approx_fn": interpolator,
    }


def cubic_spline(x, y, use_linear_interp=True, new_points=1000):
    """
    Cubic spline interpolation of a set of data points and calculate the first and second derivatives, as
    well as the point of highest curvature.
    :param x: x values
    :param y: y values
    :param use_linear_interp: whether to use linear interpolation to smooth the data first
    :param new_points: number of new points to generate for the spline
    :return:
    """
    from scipy.interpolate import CubicSpline

    x = check_sorted(x)

    if use_linear_interp:
        from scipy.interpolate import interp1d

        f = interp1d(x, y, kind="linear")
        y = f(x)

    # Create a cubic spline interpolation
    cs = CubicSpline(x, y)

    # Generate new x values for a smooth curve
    x_new = np.linspace(x.min(), x.max(), new_points)
    y_new = cs(x_new)

    # Calculate the first and second derivatives
    first_derivative = cs(x_new, 1)
    second_derivative = cs(x_new, 2)

    # Find the point of highest curvature
    curvature = compute_curvature(first_derivative, second_derivative)

    max_curvature_index = np.argmax(curvature)  # on the new x

    # retrieve in original x the point closest to the max curvature
    x_max_curvature = x_new[max_curvature_index]
    original_closest_index = np.argmin(np.abs(x - x_max_curvature))

    return {
        "x_new": x_new,
        "y_new": y_new,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
        "curv_y": curvature,
        "id_max": max_curvature_index,  # on the new x
        "id_max_origin": original_closest_index,
        "approx_fn": cs,
    }


def b_spline(
    x,
    y,
    degree=3,
    use_linear_interp=True,
    logspace=True,
    precision=1e-5,
    new_points=1000,
    **kwargs,
):
    """
    B-spline interpolation of a set of data points and calculate the first and second derivatives,
    as well as the point of highest curvature.
    :param x: x values
    :param y: y values
    :param degree: degree of the spline (default is 3, cubic)
    :param use_linear_interp: whether to use linear interpolation to smooth the data first
    :param new_points: number of new points to generate for the spline
    :param kwargs: additional arguments for the B-spline interpolation function (e.g., knots, etc.).
                  Check the documentation of make_interp_spline()
    :return:
    """
    from scipy.interpolate import make_interp_spline

    x = check_sorted(x)

    if use_linear_interp:
        from scipy.interpolate import interp1d

        f = interp1d(x, y, kind="linear")
        y = f(x)

    # Create a B-spline interpolation
    b_spl = make_interp_spline(x, y, k=degree, **kwargs)

    # Generate new x values for a smooth curve
    if logspace:
        x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), num=new_points)
    else:
        x_new = np.linspace(x.min(), x.max(), num=new_points)

    y_new = b_spl(x_new)

    # Calculate the first and second derivatives
    first_derivative = numerical_derivative(b_spl, x_new, h=precision)
    second_derivative = numerical_derivative(
        lambda x: numerical_derivative(b_spl, x, h=precision), x_new
    )
    # first_derivative = b_spl.derivative(nu=1)(x_new)
    # second_derivative = b_spl.derivative(nu=2)(x_new)

    # Find the point of highest curvature
    curvature = compute_curvature(first_derivative, second_derivative)

    max_curvature_index = np.argmax(curvature)  # on the new x

    # retrieve in original x the point closest to the max curvature
    x_max_curvature = x_new[max_curvature_index]
    original_closest_index = np.argmin(np.abs(x - x_max_curvature))

    return {
        "x_new": x_new,
        "y_new": y_new,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
        "curv_y": curvature,
        "id_max": max_curvature_index,  # on the new x
        "id_max_origin": original_closest_index,
        "approx_fn": b_spl,
    }


def monotone_interpolants(
    x, y, interpolant="pchip", logspace=True, precision=1e-5, new_points=1000
):
    """
    Compute the curvature of a monotone interpolant (e.g., PchipInterpolator or Akima1DInterpolator) and
    find the point of highest curvature.
    :param x: x values
    :param y: y values
    :param interpolant: interpolant function, pchip or akima (e.g., PchipInterpolator or Akima1DInterpolator)
    :param precision: precision of the numerical derivative
    :param new_points: number of new points to generate for the interpolant
    :return:
    """
    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

    if interpolant.lower() not in ["pchip", "akima"]:
        raise ValueError("Interpolant must be 'pchip' or 'akima'.")

    # Choose interpolator: PchipInterpolator or Akima1DInterpolator
    interpolator = (
        PchipInterpolator(x, y)
        if interpolant.lower() == "pchip"
        else Akima1DInterpolator(x, y)
    )

    # Compute curvature

    if logspace:
        x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), num=new_points)
    else:
        x_new = np.linspace(x.min(), x.max(), num=new_points)

    y_new = interpolator(x_new)
    first_derivative = numerical_derivative(interpolator, x_new, h=precision)
    second_derivative = numerical_derivative(
        lambda x: numerical_derivative(interpolator, x, h=precision), x_new
    )

    curvature = compute_curvature(first_derivative, second_derivative)

    max_curvature_index = np.argmax(curvature)  # on the new x

    # retrieve in original x the point closest to the max curvature
    x_max_curvature = x_new[max_curvature_index]
    original_closest_index = np.argmin(np.abs(x - x_max_curvature))

    return {
        "x_new": x_new,
        "y_new": y_new,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
        "curv_y": curvature,
        "id_max": max_curvature_index,  # on the new x
        "id_max_origin": original_closest_index,
        "approx_fn": interpolator,
    }


def curve_smoother(x, y, s=1.0, logspace=False, new_points=None):
    """
    Smooth a curve using B_splines, compute the first and second derivatives, as well as the point of highest curvature.
    :param x: x values
    :param y: y values
    :param s: smoothing factor
    :param new_points: number of new points to generate for the spline
    :return:
    """

    from scipy.interpolate import splev, splrep

    x = check_sorted(x)

    if new_points is None:
        x_new = x
    else:
        if logspace:
            x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), num=new_points)
        else:
            x_new = np.linspace(x.min(), x.max(), num=new_points)

    tck = splrep(x_new, y, s=s)  # default k = 3, cubic spline

    y_new = splev(x_new, tck)

    # Compute first and second derivatives
    first_derivative = splev(x_new, tck, der=1)
    second_derivative = splev(x_new, tck, der=2)

    # Compute curvature
    curvature = compute_curvature(first_derivative, second_derivative)

    max_curvature_index = np.argmax(curvature)  # on the new x

    # retrieve in original x the point closest to the max curvature
    x_max_curvature = x_new[max_curvature_index]
    original_closest_index = np.argmin(np.abs(x - x_max_curvature))

    return {
        "x_new": x_new,
        "y_new": y_new,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
        "curv_y": curvature,
        "id_max": max_curvature_index,  # on the new x
        "id_max_origin": original_closest_index,
        "approx_fn": tck,
    }


if __name__ == "__main__":
    # Example data points
    x = np.linspace(-10, 10, 100)  #
    y = 1 / (1 + np.exp(-x))  #

    approximation = monotone_interpolants(x, y, "pchip")  # cubic_spline(x, y)
    # b_spline(x,y)
    # cubic_spline(x, y)

    x_new = approximation["x_new"]
    y_new = approximation["y_new"]
    first_derivative = approximation["first_derivative"]
    second_derivative = approximation["second_derivative"]
    curvature = approximation["curv_y"]
    id_max = approximation["id_max"]
    id_max_origin = approximation["id_max_origin"]
    approx_fn = approximation["approx_fn"]

    # Plot the original data, spline, and derivatives
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o", label="Data points")
    plt.plot(x_new, y_new, label="Spline")
    plt.plot(x_new, first_derivative, "--", label="1st Derivative")
    plt.plot(x_new, second_derivative, ":", label="2nd Derivative")
    plt.plot(x_new, curvature, "-.", label="Curvature")
    plt.legend()
    plt.title("Cubic Spline and Derivatives")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_curvature_point = (x_new[id_max], approx_fn(x_new[id_max]))

    print(f"Point of highest curvature: {max_curvature_point}")

    print(x[id_max_origin], y[id_max_origin])
