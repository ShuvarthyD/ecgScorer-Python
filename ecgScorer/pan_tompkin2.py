import numpy as np
from scipy.signal import butter, filtfilt
from decimal import Decimal, ROUND_HALF_UP
import peakutils

def pan_tompkin2(ecg, fs, f1, f2):
    """
    Python version of pan_tompkin2.m

    Parameters
    ----------
    ecg : ndarray
        Input signal.
    fs : int
        The Sampling frequency in Hz.
    f1 : float
        Lower cutoff frequency in Hz.
    f2 : float
        Upper cutoff frequency in Hz.
    Returns
    -------
    qrs_amp_raw : ndarray
        Amplitudes of detected QRS peaks.
    qrs_i_raw : ndarray of str
        Indices of detected QRS peaks.
    delay : int
        Detection delay.
    
    Method
    ------
    See reference and supporting documents on ResearchGate:
    https://www.researchgate.net/publication/313673153_Matlab_Implementation_of_Pan_Tompkins_ECG_QRS_detector

    References
    ----------
    [1] Sedghamiz, H., "Matlab Implementation of Pan-Tompkins ECG QRS Detector",
        2014. Available on ResearchGate.
    [2] Pan, J., Tompkins, W. J., "A Real-Time QRS Detection Algorithm",
        IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3,
        pp. 230-236, March 1985.

    License
    -------
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Author
    ------
    Hooman Sedghamiz
    Feb 2018
    MSc. Biomedical Engineering, Linköping University
    Email: Hooman.sedghamiz@gmail.com

    Update History
    --------------
    Feb 2018:
        1. Cleaned up the code and added more comments
        2. Added to BioSigKit Toolbox
        """
    ecg = np.asarray(ecg).ravel()

    delay = 0
    skip = 0
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0

    # ---------------- Bandpass filtering ----------------
    Wn = np.array([f1, f2]) / (fs / 2)
    b, a = butter(3, Wn, btype="bandpass")
    ecg_h = filtfilt(b, a, ecg, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    ecg_h = ecg_h / np.max(np.abs(ecg_h))

    # ---------------- Derivative ----------------
    if fs != 200:
        int_c = (5 - 1) / (fs * (1/40))
        x = np.arange(1, 6)  # 1:5 in MATLAB
        y = np.array([1, 2, 0, -2, -1]) * (1/8) * fs
        xi = np.arange(1, 5, int_c)   # 1:int_c:5 in MATLAB
        b = np.interp(xi, x, y)              # interpolate
    else:
        b = np.array([1, 2, 0, -2, -1]) * (1/8) * fs

    ecg_d = filtfilt(b, 1, ecg_h, padtype = 'odd', padlen=3*(max(len(b),1)-1))
    ecg_d = ecg_d / np.max(ecg_d)

    # ---------------- Squaring ----------------
    ecg_s = ecg_d ** 2

    # ---------------- Moving average ----------------
    win = int(Decimal(0.150 * fs).quantize(0, ROUND_HALF_UP))
    ecg_m = np.convolve(ecg_s, np.ones(win) / win, mode="full")
    delay += win // 2

    # ---------------- Peak detection ----------------
    #locs, _ = find_peaks(ecg_m, distance=int(Decimal(0.2 * fs).quantize(0, ROUND_HALF_UP)))
    locs = peakutils.indexes(y=ecg_m, thres=0, min_dist=int(Decimal(0.2 * fs).quantize(0, ROUND_HALF_UP)))
    pks = ecg_m[locs]
    LLp = len(pks)

    qrs_c = np.zeros(LLp)
    qrs_i = np.zeros(LLp, dtype=int)
    qrs_i_raw = np.zeros(LLp, dtype=int)
    qrs_amp_raw = np.zeros(LLp)

    nois_c = np.zeros(LLp)
    nois_i = np.zeros(LLp, dtype=int)

    SIGL_buf = np.zeros(LLp)
    NOISL_buf = np.zeros(LLp)
    SIGL_buf1 = np.zeros(LLp)
    NOISL_buf1 = np.zeros(LLp)
    THRS_buf = np.zeros(LLp)
    THRS_buf1 = np.zeros(LLp)

    THR_SIG = np.max(ecg_m[: 3 * fs]) / 3
    THR_NOISE = np.mean(ecg_m[: 3 * fs]) / 2
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    THR_SIG1 = np.max(ecg_h[: 2 * fs]) / 3
    THR_NOISE1 = np.mean(ecg_h[: 2 * fs]) / 2
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    Beat_C = 0
    Beat_C1 = 0
    Noise_Count = 0

    for i in range(LLp):
        if locs[i] - win >= 0 and locs[i] < len(ecg_h):
            seg = ecg_h[locs[i] - win : (locs[i] + 1)]
            y_i = np.max(seg)
            x_i = np.argmax(seg)
        else:
            seg = ecg_h[max(0, locs[i] - win) : (locs[i] + 1)]
            y_i = np.max(seg)
            x_i = np.argmax(seg)

        if Beat_C >= 9:
            diffRR = np.diff(qrs_i[(Beat_C - 1) - 8 : Beat_C])
            mean_RR = np.mean(diffRR)
            comp = qrs_i[Beat_C - 1] - qrs_i[Beat_C - 2]
            if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                THR_SIG *= 0.5
                THR_SIG1 *= 0.5
            else:
                m_selected_RR = mean_RR

        test_m = m_selected_RR or mean_RR or 0

        if test_m and Beat_C > 0:
            if (locs[i] - qrs_i[Beat_C - 1]) >= int(Decimal(1.66 * test_m).quantize(0, ROUND_HALF_UP)):
                start = qrs_i[Beat_C - 1] + int(Decimal(0.2 * fs).quantize(0, ROUND_HALF_UP))
                end = (locs[i] + 1) -int(Decimal(0.2 * fs).quantize(0, ROUND_HALF_UP))
                if end > start:
                    pks_temp = np.max(ecg_m[start:end])
                    locs_temp = start + np.argmax(ecg_m[start:end])
                    if pks_temp > THR_NOISE:
                        Beat_C += 1
                        qrs_c[Beat_C - 1] = pks_temp
                        qrs_i[Beat_C - 1] = locs_temp
                        seg2 = ecg_h[max(0, locs_temp - win) : locs_temp + 1]
                        y_i_t = np.max(seg2)
                        x_i_t = np.argmax(seg2)
                        if y_i_t > THR_NOISE1:
                            Beat_C1 += 1
                            qrs_i_raw[Beat_C1 - 1] = locs_temp - win + x_i_t
                            qrs_amp_raw[Beat_C1 - 1] = y_i_t
                            SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1
                        SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV

        if pks[i] >= THR_SIG:
            if Beat_C >= 3:
                if (locs[i] - qrs_i[Beat_C - 1]) <= int(Decimal(0.36 * fs).quantize(0, ROUND_HALF_UP)):
                    s1 = np.mean(np.diff(ecg_m[locs[i] - int(Decimal(0.075 * fs).quantize(0, ROUND_HALF_UP)) : (locs[i] + 1) ]))
                    s2 = np.mean(np.diff(ecg_m[qrs_i[Beat_C - 1]- int(Decimal(0.075 * fs).quantize(0, ROUND_HALF_UP)) : qrs_i[Beat_C - 1] + 1]))
                    if abs(s1) <= 0.5 * abs(s2):
                        Noise_Count += 1
                        nois_c[Noise_Count - 1] = pks[i]
                        nois_i[Noise_Count - 1] = locs[i]
                        skip = 1
                        NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                        NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

            if skip == 0:
                Beat_C += 1
                qrs_c[Beat_C - 1] = pks[i]
                qrs_i[Beat_C - 1] = locs[i]
                if y_i >= THR_SIG1:
                    Beat_C1 += 1
                    qrs_i_raw[Beat_C1 - 1] = locs[i] - win + x_i
                    qrs_amp_raw[Beat_C1 - 1] = y_i
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1
                SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV

        elif THR_NOISE <= pks[i] < THR_SIG:
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        else:
            Noise_Count += 1
            nois_c[Noise_Count - 1] = pks[i]
            nois_i[Noise_Count - 1] = locs[i]
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        if NOISE_LEV != 0 or SIG_LEV != 0:
            THR_SIG = NOISE_LEV + 0.25 * abs(SIG_LEV - NOISE_LEV)
            THR_NOISE = 0.5 * THR_SIG

        if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
            THR_SIG1 = NOISE_LEV1 + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
            THR_NOISE1 = 0.5 * THR_SIG1

        SIGL_buf[i] = SIG_LEV
        NOISL_buf[i] = NOISE_LEV
        THRS_buf[i] = THR_SIG
        SIGL_buf1[i] = SIG_LEV1
        NOISL_buf1[i] = NOISE_LEV1
        THRS_buf1[i] = THR_SIG1

        skip = 0
        ser_back = 0

    qrs_i_raw = qrs_i_raw[:Beat_C1]
    qrs_amp_raw = qrs_amp_raw[:Beat_C1]

    return qrs_amp_raw, qrs_i_raw, delay