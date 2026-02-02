import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def isFlatline(sig, Fs):
    """
    Python version of isFlatline.m

    Parameters
    ----------
    sig : ndarray
        Input signal.
    Fs : float
        The Sampling frequency in Hz.

    Returns
    -------
    condition : int
        1 if flatline detected, else 0.
    """

    sig = np.asarray(sig)

    # All-zero signal
    if np.sum(sig) == 0:
        return 1

    # normalize(sig,'range',[0,50]) + fix()
    sig_min = sig.min()
    sig_max = sig.max()
    if sig_max > sig_min:
        sig = (sig - sig_min) / (sig_max - sig_min) * 50
    else:
        sig = np.zeros_like(sig)

    sig = np.fix(sig)

    condition1 = flat_calc(sig, Fs, c=0)

    sigo = sig[sig > (1.5 * np.mean(sig))]
    condition11 = flat_calc(sigo, Fs, c=1)

    condition = int(condition1 or condition11)
    return condition


def flat_calc(sig, Fs, c):
    """
    Helper function for flatline detection
    """

    if len(sig) < 3:
        return 0

    xx = np.diff(sig)

    flat_i = np.where((xx == 0) | (xx == -1) | (xx == 1))[0]

    if len(flat_i) < 3:
        return 0

    flato = np.where(np.diff(flat_i, 2) == 0)[0]

    k = 0
    matrix = []

    while k < len(flato) - 1:
        counter = 0

        while flato[k + 1] == flato[k] + 1:
            counter += 1
            k += 1
            if k == len(flato) - 1:
                break

        matrix.append(counter)
        k += 1

        if k >= len(flato) - 2:
            break

    if len(matrix) == 0:
        return 0

    if c == 1:
        flaty = np.sum(matrix)
        return int(flaty > int(Decimal(2.5 * Fs).quantize(0, ROUND_HALF_UP)))
    else:
        flaty = np.max(matrix)
        return int(flaty > int(Decimal(1.5 * Fs).quantize(0, ROUND_HALF_UP)))

