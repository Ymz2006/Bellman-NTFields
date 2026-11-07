"""Plain sin / cos rows at the environment's own frequencies: the 128 largest peaks of the 2-D power spectrum of the obstacle
signal (half plane, at least one box harmonic apart). This is 'choose B from the environment' with ntrl-demo's form of
feature, [sin 2 pi B q, cos 2 pi B q], and no fitted amplitudes."""
import numpy as np


def peaks(o, count=128, pad=2):
    """(count, 2) frequencies (cycles per normalized unit) of the largest peaks of |FFT(o - mean)|^2, one per conjugate pair."""
    R = o.shape[0]
    size = pad * R
    F = np.fft.fftshift(np.abs(np.fft.fft2(o - o.mean(), s=(size, size))) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(size, 1.0 / (R - 1)))
    half = np.ones_like(F, bool)
    half[:size // 2] = False                          # f_x >= 0 keeps one of each conjugate pair
    half[size // 2, :size // 2] = False               # f_x = 0: f_y > 0
    half[size // 2, size // 2] = False                # the mean
    taken = np.zeros_like(F, bool)
    out = []
    for i in np.argsort(-(F * half).ravel()):
        a, b = divmod(int(i), size)
        if not half[a, b] or taken[a, b]:
            continue
        out.append((f[a], f[b]))
        taken[max(0, a - pad + 1):a + pad, max(0, b - pad + 1):b + pad] = True
        if len(out) == count:
            break
    return np.array(out)
