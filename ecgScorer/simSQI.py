import numpy as np
from scipy.signal import butter, filtfilt
import pan_tompkin2
from decimal import Decimal, ROUND_HALF_UP


def simSQI(sig, i2, flag, fs):
    """
    Python version of simSQI.m

    Parameters
    ----------
    sig : ndarray
        ECG signal
    i2 : ndarray
        R-peak indices (used if flag == 1)
    flag : int
        If 1, use provided R-peaks; otherwise detect them
    fs : float
        Sampling frequency

    Returns
    -------
    iScore : float
        Instantaneous SQI
    aScore : float
        Averaged SQI
    """

    sig = np.asarray(sig)

    if flag == 1:
        R = np.asarray(i2, dtype=int)

    else:
        # Bandpass filter 1–40 Hz
        b, a = butter(3, [1, 40], btype='bandpass', fs=fs)
        sig = filtfilt(b, a, sig, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

        # R-peak detection (external function)
        _, R, _ = pan_tompkin2.pan_tompkin2(sig, fs, 8, 20)

        if len(R) < 4:
            _, R, _ = pan_tompkin2.pan_tompkin2(sig, fs, 5, 15)

        #R = np.asarray(R, dtype=int)

    iScore, aScore = runSimSQI(sig, R)
    return iScore, aScore


def runSimSQI(sig, R):
    sig = np.asarray(sig)
    R = np.asarray(R, dtype=int)

    # Weighting functions
    Pgroup = lambda x: 1.7 * np.exp(-0.52 * x) - 0.005
    Pnoise = lambda x: 1.6 * np.exp(-0.38 * x) - 0.01

    nb_B = len(R)

    rr = np.diff(R)
    mRR = min(np.mean(rr), np.median(rr))
    #med_R = int(round(mRR))
    med_R = int(Decimal(mRR).quantize(0, ROUND_HALF_UP))

    #beat_on = np.round(R - (med_R - 1) / 2).astype(int)
    #beat_off = np.round(R + (med_R - 1) / 2).astype(int)
    
    # Shifted values
    R_shifted_minus = R - (med_R - 1)/2
    R_shifted_plus  = R + (med_R - 1)/2
    # MATLAB-style rounding using Decimal, element-wise
    beat_on  = np.array([int(Decimal(x).quantize(0, ROUND_HALF_UP)) for x in R_shifted_minus])
    beat_off = np.array([int(Decimal(x).quantize(0, ROUND_HALF_UP)) for x in R_shifted_plus])

    # Border checks
    if beat_on[0] < 0:
        a = np.where(beat_on < 0)[0][-1] + 1
    else:
        a = 0

    if beat_off[-1] >= len(sig):
        z = np.where(beat_off >= len(sig))[0][0] - 1
    else:
        z = nb_B - 1

    # Beat matrix
    Nbeats = z - a + 1
    Beats = np.zeros((Nbeats, med_R))

    ii = 0
    for i in range(a, z + 1):  # range(a, z+1) because Python is exclusive
        #print(R, beat_on, beat_off)
        #start = beat_on[i]
        #end = start + med_R
        start = beat_on[i]
        end = beat_off[i]
        Beats[ii, :] = sig[start:end + 1]
        ii += 1

    # Correlation matrix
    corMat = np.corrcoef(Beats)
    corMat_mean = np.mean(corMat, axis=0)

    iScore = np.mean(corMat_mean)

    # Similarity check
    thr1 = 0.65
    thr2 = 0.75

    idx_maxi = np.argmax(corMat_mean)
    spB = corMat_mean[idx_maxi]

    idx_worst = np.where(corMat[idx_maxi, :] < thr1)[0]
    idx_good = np.where((corMat[idx_maxi, :] >= thr1) & (np.abs(corMat[idx_maxi, :] - 1) > 1e-12))[0]

    score_wg = []
    noise_count = 0
    group = 0

    L = len(idx_worst)

    if L == 1:
        noise_count = 1

    else:
        count = 0
        while count < L and len(idx_worst) > 0:
            idx_res = corMat[idx_worst[0], idx_worst] > thr2
            idx_res = idx_worst[idx_res]

            if len(idx_res) == 1:
                noise_count += 1
                cc = corMat_mean[idx_res[0]]
                score_wg.append(cc * Pnoise(noise_count))
                idx_worst = idx_worst[1:]
                count += 1
                continue

            comp = np.isin(idx_worst, idx_res)
            b_res = np.where(comp)[0]

            group += 1
            sw = np.mean(corMat[np.ix_(idx_worst[b_res], idx_worst[b_res])])
            score_wg.append(sw * Pgroup(group))

            idx_worst = idx_worst[~comp]
            count += 1

    # Final score
    Ngood = len(idx_good) + 1

    if Ngood == 1:
        score_gg = spB
    elif Ngood == 2:
        score_gg = (spB + corMat[idx_maxi, idx_good[0]]) / 2
    else:
        idx_all = np.concatenate(([idx_maxi], idx_good))
        score_gg = np.mean(corMat[np.ix_(idx_all, idx_all)])

    if len(score_wg) > 0:
        aScore = (np.mean(score_wg) + score_gg) / 2
    else:
        aScore = score_gg

    return iScore, aScore