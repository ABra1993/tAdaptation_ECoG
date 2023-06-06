import numpy as np
import math

def gammaPDF(t, tau, n):
    """ Returns values of a gamma function for a given timeseries.

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints
    tau : float
        peak time
    n : int
        effects response decrease after peak

    returns
    -----------------------
    y_norm : array dim(1, T)
        contains gamma values for each timepoint
    """

    y = (t/tau)**(n - 1) * np.exp(-t/tau) / (tau * math.factorial(n - 1))
    y_norm = y/np.sum(y)

    return y_norm

def exponential_decay(t, tau):
    """ Impulse Response Function

    params
    -----------------------
    timepots : int
        length of timeseries
    tau : float
        peak time

    returns
    -----------------------
    y_norm : array dim(1, T)
        contains value for each timepoint
    """

    y = np.exp(-t/tau)
    y_norm = y/np.sum(y)

    return y_norm