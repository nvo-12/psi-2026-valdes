import numpy as np
import pytest

from psi_lab.senales import cuadrada, espectro


def test_cuadrada_solo_tiene_armonicos_impares():
    f, fs = 50, 1000
    _, x = cuadrada(f=f, fs=fs, duracion=1)
    freqs, amp = espectro(x, fs)

    def amplitud_en(frecuencia):
        return amp[np.argmin(np.abs(freqs - frecuencia))]

    for impar in (f, 3 * f, 5 * f):
        assert amplitud_en(impar) > 0.1
    for par in (2 * f, 4 * f, 6 * f):
        assert amplitud_en(par) < 1e-6
    # Serie de Fourier: el fundamental de una cuadrada de amplitud 1 es 4/pi
    assert amplitud_en(f) == pytest.approx(4 / np.pi, rel=0.02)
