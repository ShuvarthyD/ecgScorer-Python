import numpy as np
from scipy.signal import butter, filtfilt


def isPurenoise(v, fs):
    """
    Python version of isPurenoise.m

    Parameters
    ----------
    v : ndarray
        Input signal.
    fs : float
        The Sampling frequency in Hz.

    Returns
    -------
    nzc : int
        Number of zero crossings after filtering.
    """

    v = np.asarray(v)
    b, a = butter(3, 0.5, btype='high', fs=fs)
    v = filtfilt(b, a, v, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    v_min = v.min()
    v_max = v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min) * 2 - 1
    else:
        v = np.zeros_like(v)
    zci = np.where(np.diff(np.sign(v)) != 0)[0]
    nzc = len(zci)

    return nzc
