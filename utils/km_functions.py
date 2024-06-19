# Library for calculation Kramers-Moyal coefficients
from kramersmoyal import km

import numpy as np
from scipy.optimize import curve_fit


def km_get_drift(data, bw, delta_t):
    """
    Calculate the drift coefficient using the first Kramers-Moyal coefficient

    :param bw: bandwidth parameter
    :param data: angular velocity data without any gaps
    :param delta_t: time step resolution of the data
    :return: drift and space of km. Drift coefficients are in drift[1]
    """
    bins = np.array([6000])
    powers = [0, 1]

    # get drift coefficient
    drift, space = km(data, powers=powers, bins=bins, bw=bw)

    # normalize drift
    drift = drift / delta_t

    return drift, space


def km_get_primary_control(drift, space, offset=500):
    """
    Calculate the primary control/drift as the slope of the drift coefficient
    """

    # find zero frequency
    zero_frequency = np.argmin(space[0] ** 2)

    l_offset = offset
    r_offset = offset
    if zero_frequency < l_offset:
        l_offset = zero_frequency
        print(f'left offset modified: {l_offset}, zero frequency: {zero_frequency}')
    if zero_frequency > 6000 - r_offset:
        r_offset = 6000 - zero_frequency
        print(f'right offset modified: {r_offset}, zero frequency: {zero_frequency}')

    # Get slope of drift term which is c_1
    c_1 = curve_fit(lambda t, a, b: a - b * t,
                    space[0][zero_frequency - l_offset:zero_frequency + r_offset],
                    drift[1][zero_frequency - l_offset:zero_frequency + r_offset],
                    p0=(0.00002, 0.005),
                    maxfev=10000)[0][1]

    return c_1


def km_get_diffusion(data, bw, delta_t):
    """
    Calculate the second Kramers-Moyal coefficient (diffusion)

    :param bw: bandwidth parameter
    :param data: angular velocity data without any gaps
    :param delta_t: time step resolution of the data
    """
    bins = np.array([6000])
    powers = [0, 2]

    # get diffusion coefficient
    diffusion, space = km(data, powers=powers, bins=bins, bw=bw)

    # normalize diffusion
    diffusion = diffusion / delta_t

    # find zero frequency
    zero_frequency = np.argmin(space[0] ** 2)

    # evaluate diffusion at zero frequency and get epsilon
    return diffusion[1, zero_frequency]
