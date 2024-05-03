from dataclasses import dataclass

import numpy as np
import pywt
from scipy.ndimage import convolve1d

from eitprocessing.filters import TimeDomainFilter


@dataclass(kw_only=True)
class MODWTFilter(TimeDomainFilter):
    """Maximum Overlap Digital Wavelet Transform filter for filtering in the time domain."""

    wavelet: str = "sym13"
    sample_frequency: float
    decomposition_level: int | None = None

    def apply_filter(self, input_data: np.ndarray) -> np.ndarray:
        """Generates MODWT filtered version of original data."""
        decomposition_level = self.decomposition_level
        if not self.decomposition_level:
            decomposition_level = 5 if self.sample_frequency > 30 else 4

        coefs_modwt = _modwt(input_data, self.wavelet, decomposition_level)
        coefs_modwt[0 : decomposition_level - 1] *= 0
        return _imodwt(coefs_modwt, self.wavelet)


def _upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def _period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def _circular_convolve_mra(h_j_o, w_j):
    """Calculate the mra D_j."""
    return convolve1d(w_j, np.flip(h_j_o), mode="wrap", origin=(len(h_j_o) - 1) // 2)


def _circular_convolve_d(h_t, v_j_1, j):
    """
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j).
    """
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2 ** (j - 1))

    # make kernel
    for i, h in enumerate(h_t):
        ker[i * 2 ** (j - 1)] = h

    w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
    return w_j


def _circular_convolve_s(h_t, g_t, w_j, v_j, j):
    """
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d.
    """
    N = len(v_j)

    h_ker = np.zeros(len(h_t) * 2 ** (j - 1))
    g_ker = np.zeros(len(g_t) * 2 ** (j - 1))

    for i, (h, g) in enumerate(zip(h_t, g_t)):
        h_ker[i * 2 ** (j - 1)] = h
        g_ker[i * 2 ** (j - 1)] = g

    # v_j_1 = np.zeros(N)

    v_j_1 = convolve1d(
        w_j,
        np.flip(h_ker),
        mode="wrap",
        origin=(len(h_ker) - 1) // 2,
    ).astype(np.float64)
    v_j_1 += convolve1d(
        v_j,
        np.flip(g_ker),
        mode="wrap",
        origin=(len(g_ker) - 1) // 2,
    ).astype(np.float64)
    return v_j_1


def _modwt(x, filters, level):
    """
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab.
    """
    # filter
    wavelet = pywt.Wavelet(filters)  # type: ignore
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = _circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = _circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def _imodwt(w, filters):
    """Inverse modwt."""
    # filter
    wavelet = pywt.Wavelet(filters)  # type: ignore
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = _circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def _modwtmra(w, filters):
    """Multiresolution analysis based on MODWT."""
    # filter
    wavelet = pywt.Wavelet(filters)  # type: ignore
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = _upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = _upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2)
        h_j_t_o = _period_list(h_j_t, N)
        D.append(_circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = _upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.0))
    g_j_t_o = _period_list(g_j_t, N)
    S = _circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)
