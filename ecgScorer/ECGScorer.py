import numpy as np
import isFlatline, isPurenoise, pan_tompkin2, simSQI

def scorer12(x, fs):
    """
    Python version of scorer12.m

    Parameters
    ----------
    x : ndarray
        Single or multichannel ECG signal. The channel must be a column vector.
    fs : float
        The Sampling frequency in Hz.

    Returns
    -------
    label : ndarray
        The quality class of the ECG. 3 possible values : 0 = Good Quality Signal,
        0.5 = intermediate Quality signal, 1 = Bad Quality Signal.
    comment : ndarray of str
        String describibg the problem of the signal or the number of passage
        to compute the aScore.
    signalNum : ndarray
        The order of signal in the provided database
    aScore : ndarray
        Adjusted ECG quality Score between 0 and 1.
    iScore : ndarray
        Initial ECG Quality Score between 0 and 1. When iScore/aScore = -1 the 
        signal was detected to be a pure noise or saturated/flatline signal.
    """

    Fs = fs
    numberOfSig = x.shape[1]

    aScore = -np.ones(numberOfSig)
    iScore = -np.ones(numberOfSig)
    label = np.random.rand(numberOfSig)
    comment = np.array([""] * numberOfSig, dtype=object)
    signalNum = np.arange(1, numberOfSig + 1)

    for i in range(numberOfSig):
        sig = x[:, i].copy()

        # 1) Flat line / saturation detection
        flaty = isFlatline.isFlatline(sig, fs)

        if flaty == 1:
            print("*******************************************************")
            print("| flat line / saturation / electrode problem detected |")
            print("*******************************************************")
            label[i] = 1
            comment[i] = "flat line / saturation / electrode problem detected"
            continue

        # 2) Pure noise detection
        nzc = isPurenoise.isPurenoise(sig, fs)

        if nzc >= 200:
            print("***********************")
            print("| pure noise detected |")
            print("***********************")
            label[i] = 1
            comment[i] = "pure noise detected"
            continue

        if 160 <= nzc < 200:
            print("************************")
            print("| suspected pure noise |")
            print("************************")
            comment[i] = "suspected pure noise"

        # Pan Tompkins detector doubled band pass sqi
        sig_min = sig.min()
        sig_max = sig.max()
        if sig_max > sig_min:
            sig = (sig - sig_min) / (sig_max - sig_min)
        else:
            sig = np.zeros_like(sig)

        qrs2, i2, _ = pan_tompkin2.pan_tompkin2(sig, fs, 8, 20)

        N2 = len(i2)
        mm = N2
        tmSQI = np.max(np.diff(np.concatenate(([0], i2, [10 * fs]))))
        flag = 0

        # Heuristic rules for bad signals
        if mm <= 3 or tmSQI > 2.5 * fs:
            print("found less than 4 peaks")
            print("---- Search the R peaks using the reversed signal ----")

            sig_rev = sig[::-1]
            _, irev, _ = pan_tompkin2.pan_tompkin2(sig_rev, fs, 8, 20)

            df = np.concatenate(([0], irev, [10 * fs]))
            diffs = np.diff(df)
            maxi = np.max(diffs)
            imax = np.argmax(diffs)

            if len(irev) <= 3:
                print("** the number of beats is truly less than 4 **")
                label[i] = 1
                comment[i] = "HR found less than 24 BPM"
                continue
            else:
                i2 = len(sig) - irev
                i2 = i2[::-1]
                qrs2 = sig[i2]
                flag = 1
                print("** missed peaks recovered **")

            if maxi >= 3 * fs:
                _, irev2, _ = pan_tompkin2.pan_tompkin2(
                    sig_rev[df[imax] + 10 :], fs, 8, 20
                )
                irev2 = irev2 + df[imax] + 10 - 1
                irev = np.concatenate((irev[: imax], irev2))

                if np.max(np.diff(np.concatenate(([0], irev, [10 * fs])))) > 2.5 * fs:
                    label[i] = 1
                    comment[i] = "time between two beat surpassed 2.5 sec"
                    print("time between two beat surpassed 2.5 sec")
                    continue
                else:
                    i2 = len(sig) - irev
                    i2 = i2[::-1]
                    qrs2 = sig[i2]
                    flag = 1
                    print("found r-peaks in the 2.5 sec interval")

        QRS_amp = np.max(np.diff(qrs2)) if len(qrs2) > 1 else 0

        if QRS_amp > 0.8:
            print("*****************")
            print("| abrupt Change |")
            print("*****************")
            label[i] = 1
            comment[i] = "abrupt Change detected"
            continue

        # Beats' average correlation algorithm
        L2 = 0.35
        L3 = 0.375

        iscore, ascore = simSQI.simSQI(sig, i2, flag, Fs)
        iScore[i] = iscore
        aScore[i] = ascore

        if iscore < L2:
            label[i] = 1
            print("********** Bad Signal *************")
            print(f"the iscore {iscore} is < {L2}")
            continue

        elif iscore >= 0.9:
            comment[i] = "one pass"
            label[i] = 0

        elif L2 <= iscore < 0.9 and ascore >= 0.9:
            comment[i] = "two pass"
            label[i] = 0

        elif L2 <= iscore < 0.9 and L3 <= ascore < 0.9:
            label[i] = 0.5

        else:
            label[i] = 1

        if label[i] == 1:
            print("********** Bad Signal *************")
            print(f"the iscore {iscore} the aScore is {ascore}\n")
        else:
            print("********** Good Signal *************")
            print(f"the iscore {iscore} the aScore is {ascore}\n")

    return label, comment, signalNum, aScore, iScore