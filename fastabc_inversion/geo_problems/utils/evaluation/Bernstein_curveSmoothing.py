import math

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.distributions.empirical_distribution as emp_dist

# based on : Application of Bernstein Polynomials for smooth estimation
# of a distribution and density function
# http://personal.psu.edu/gjb6/mypdfpap/2002JSPI105.pdf


def transform_x(x, min_x, max_x):
    """
    Transform x into [0,1]
    :param x: locations to transform
    :return: transformed x (sorted)
    """
    transformed_x = (x - min_x) / (max_x - min_x)
    return transformed_x


def bpoly(m, k, t):
    """
    Evaluation Bernstein polynomial
    :param m: maximum degree of the polynomial
    :param k: degree of the basis function to evaluate
    :param t: location in [0,1]
    :return: Bernstein polynomial of degree m evaluated at t in [0,1]
    """
    binom = math.comb(m, k)
    b_poly = binom * ((1 - t) ** (m - k)) * (t ** (k))
    return b_poly


def approx_fn(f, m, bpoly, t, min_x, max_x):
    """
    Smooth a given function with Bernstein polynimials
    :param f: the function (adapted to cases where f is a linear interpolation between points for example)
    :param m: degree of the polynomials
    :param bpoly: the function to evaluate the Bernstein polynomials
    :param t: the location where to evaluate the polynomail , should be in [0,1]
    :param min_x: minimum value of x
    :param max_x: minimum value of y
    :return: smoothed estimation of f at t
    """
    sum = 0
    for k in range(m + 1):
        # v = (k/m)*(max_x - min_x) + min_x
        sum = sum + f(k / m) * bpoly(m, k, t)
        # = sum + f(v) * bpoly(m, k, t)

    return sum


def deriv1_approx_fn(f, m, bpoly, t, min_x, max_x):
    """
    Estimate the first derivative of a Bernstein smoothed function
    :param f: the function (adapted to cases where f is a linear interpolation between points for example)
    :param m: degree of the polynomials
    :param bpoly: the function to evaluate the Bernstein polynomials
    :param t: the location where to evaluate the polynomail , should be in [0,1]
    :param min_x: minimum value of x
    :param max_x: minimum value of y
    :return: first derivative evaluated at t
    """
    sum = 0

    for k in range(m):
        # v1 = (k / m) * (max_x - min_x) + min_x
        # v2 = ((k+1)/m) * (max_x - min_x) + min_x

        # sum = sum + (f(v2) - f(v1)) * bpoly(m-1, k, t) #/ max_x
        dx = max_x - min_x
        sum = sum + (f((k + 1) / m) - f(k / m)) * bpoly(m - 1, k, t) / dx

    return sum * m


def deriv2_approx_fn(f, m, bpoly, t, min_x, max_x):
    """
    Estimate the second derivative of a Bernstein smoothed function
    :param f: the function (adapted to cases where f is a linear interpolation between points for example)
    :param m: degree of the polynomials
    :param bpoly: the function to evaluate the Bernstein polynomials
    :param t: the location where to evaluate the polynomial , should be in [0,1]
    :param min_x: minimum value of x
    :param max_x: minimum value of y
    :return: second derivative evaluated at t
    """
    sum = 0

    for k in range(m - 1):
        # v1 = (k / m) * (max_x - min_x) + min_x
        # v2 = ((k + 1) / m) * (max_x - min_x) + min_x
        # v3 = ((k + 2) / m) * (max_x - min_x) + min_x

        # sum = sum + (f(v1) - 2*f(v2) +f(v3)) * bpoly(m - 2, k, t) #/ (max_x ** 2)
        dx1 = max_x - min_x
        dx2 = max_x - min_x
        sum = sum + (f(k / m) - 2 * f((k + 1) / m) + f((k + 2) / m)) * bpoly(
            m - 2, k, t
        ) / (dx1 * dx2)
    return sum * m * (m - 1)


def curvature(t, deriv1, deriv2, f, m, bpoly, min_x, max_x):
    d1 = deriv1(f, m, bpoly, t, min_x, max_x)
    d2 = deriv2(f, m, bpoly, t, min_x, max_x)

    return np.abs(d2) / ((1 + d1**2) ** (3 / 2))


if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BPoly, UnivariateSpline, interp1d
    from scipy.misc import derivative

    # x = np.random.randn(100)
    # ecdf = emp_dist.ECDF(x)
    # y = np.log(ecdf(np.sort(x)))
    x = np.linspace(0, 5, 5)
    y = x**2  # np.exp(x) #3 * x + 2
    min_x = np.min(x)
    max_x = np.max(x)
    x_new = np.linspace(min_x, max_x, 500)

    x_t = transform_x(x, min_x, max_x)
    f = interp1d(x_t, y)

    """
    m = 3
    x_k = []
    for i in range(x.size-1):
        k_i = []
        for j in range(m+2):
            delta = x[i+1] - x[i]
            k_i.append(x[i] + delta / (m+1)*j)
        x_k.append(np.array(k_i))

    x_k = np.array(x_k)
    c = f(x_k).T

    #f_bpoly = BPoly(y.reshape(-1, 1), [min_x, max_x])
    f_bpoly = BPoly(c, x)
    y_new = f_bpoly(x_new)

    plt.plot(x, y, 'o', x_new, y_new, '-')
    plt.show()
    plt.close()

    f_bpoly_d1 = np.diff(y_new)/np.diff(x_new)
    f_bpoly_d2 = np.diff(f_bpoly_d1)/np.diff(x_new[1:])

    plt.plot(x_new[1:], f_bpoly_d1, '-', x_new[2:], f_bpoly_d2, '--')
    plt.show()
    plt.close()
    """
    t = transform_x(x_new, min_x, max_x)
    # t = np.linspace(0,1, 200)
    m = 100

    f_bm_eval = []
    f_1d_bm_eval = []
    f_2d_bm_eval = []
    curv_eval = []
    for t_i in t:
        f_bm_eval.append(approx_fn(f, m, bpoly, t_i, min_x, max_x))  # smoothed function
        f_1d_bm_eval.append(deriv1_approx_fn(f, m, bpoly, t_i, min_x, max_x))
        f_2d_bm_eval.append(deriv2_approx_fn(f, m, bpoly, t_i, min_x, max_x))
        curv_eval.append(
            curvature(
                t_i, deriv1_approx_fn, deriv2_approx_fn, f, m, bpoly, min_x, max_x
            )
        )

    plt.plot(x, y, "o", x_new, f(t), "-", x_new, f_bm_eval, ":")
    plt.plot(x_new, f_1d_bm_eval)
    plt.plot(x_new, f_2d_bm_eval)
    plt.show()
    plt.close()

    plt.plot(x, y, "o", x_new, f(t), "-", x_new, f_bm_eval, ":")
    plt.show()
    plt.close()

    plt.plot(x_new, f_1d_bm_eval)
    plt.show()
    plt.close()

    plt.plot(x_new, f_2d_bm_eval)
    plt.show()
    plt.close()

    plt.plot(x_new, curv_eval)
    plt.show()
    plt.close()


# %%
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import interpolate


#y = np.array([1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 3.023333333333334e-06, 0.00011410000000000002, 0.00023900000000000006, 0.0004723333333333335, 0.001335666666666667, 0.0018666666666666669, 0.002693333333333334, 0.005073333333333335, 0.0071533333333333345, 0.008590000000000002, 0.009156666666666669, 0.0152, 0.016533333333333334, 0.018966666666666666, 0.021233333333333337, 0.024066666666666667, 0.02946666666666667, 0.0341, 0.04493333333333333, 0.05363333333333334, 0.05773333333333334, 0.07616666666666667, 0.11066666666666665, 0.15166666666666664, 0.21233333333333335, 0.5406666666666667, 0.7813333333333334, 0.915, 0.9953333333333333, 1.0, 1.0, 1.0])
#x = np.array([21.52074731, 21.66597741, 21.49996929, 21.60537033, 21.55939178, 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100., 105., 110., 120., 130., 140., 150., 180., 210.,  250.,  500.,  750., 1000., 1500., 2000., 2500., 3000.])
x = np.array([ 574.9655599 ,  573.22950439,  576.76206462,  577.57817383,
        577.01323649,  580.92140503,  574.83307699,  578.24289754,
        576.24145711,  576.25827637,  572.91430054,  576.73215332,
        573.9568278 ,  575.45913289,  577.34394735,  573.10018921,
        574.30634359,  574.393455  ,  576.5668335 ,  574.27024943,
        573.76526286,  575.91904907,  574.68332113,  575.39362386,
        600.        ,  640.        ,  680.        ,  720.        ,
        760.        ,  800.        ,  840.        ,  880.        ,
        920.        ,  960.        , 1000.        , 1250.        ,
       1500.        , 2000.        , 2500.        , 3000.       ])
y = np.array([1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 1.0000000000000025e-50, 2.4933333333333344e-05, 0.027766666666666672, 0.169, 0.3416666666666666, 0.541, 0.6863333333333334, 0.783, 0.8649999999999999, 0.9223333333333334, 0.9516666666666667, 0.9696666666666666, 1.0, 1.0, 1.0, 1.0, 1.0])

y = y[:-10]
x = x[:-10]
LOG = True
if LOG is True:
    y = np.log(y)
    #x = np.log(x)


fig = plt.figure(dpi=300)
# plt.scatter(xnew, ynew, s=0.1)
plt.plot(x, y, 'o')

plt.show()


#f = interpolate.interp1d(x, y, kind="quadratic")

#xnew = np.arange(x[0], x[-1], 1)
#ynew = f(xnew)   # use interpolation function returned by `interp1d`


# f = np.poly1d(np.polyfit(xnew, ynew, 3))
# xnew = np.arange(21.52074731, 3000.0, 0.1)
# ynew = f(xnew)

#fig = plt.figure(dpi=300)
# plt.scatter(xnew, ynew, s=0.1)
#plt.plot(x, y, 'o', xnew, ynew, '-')

#plt.show()



# x = np.linspace(x_min, x_max, 100)
# y = norm.pdf(x, mean, std)

# 1st inflection point estimation
dy = np.diff(y) / np.diff(x)  # first derivative
# ddy = np.diff(dy) / np.diff(xnew[1:])  # second derivative
# idx_max_dy = np.argmax(ddy)
#print(idx_max_dy)

# windowing
prev_i = 0
keep_i_index = 0
for i in range(dy.size):
    if dy[i+1] < dy[i]/2:
        keep_i_index = i+1
        break

keep_i_index += 1 # we take the following value to be on the safe side
print("x value at keep_i_index", x[keep_i_index])


fig = plt.figure(dpi=500)
plt.plot(x[1:], dy, color='blue')
plt.plot(x[keep_i_index+1], dy[keep_i_index], 'or', label='estimated inflection point')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
plt.close()

fig = plt.figure(dpi=500)
plt.plot(x, y, color='green')
plt.plot(x[keep_i_index], y[keep_i_index], 'or', label='estimated inflection point')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()
plt.close()


"""
