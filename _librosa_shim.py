"""Stand-in for the one librosa function parakeet-mlx uses (librosa.filters.mel, Slaney scale).

The real librosa pulls numba + llvmlite + scikit-learn into the packaged app (about 140 MB) for a
single mel filter bank. The packaged build excludes them and calls install() at start-up. Outputs
match librosa 1.0 (tests/test_units.py compares them when librosa is importable)."""

import sys
import types

import numpy as np

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=float)
    mel = f / _F_SP
    log_region = f >= _MIN_LOG_HZ
    mel = np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOGSTEP, mel)
    return mel


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=float)
    f = _F_SP * m
    log_region = m >= _MIN_LOG_MEL
    return np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), f)


def mel(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=np.float32):
    if htk or norm != "slaney":
        raise NotImplementedError("only the default Slaney mel scale is available in the packaged app")
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=dtype)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


def install() -> None:
    """Make `import librosa` resolve to this stand-in (only when the real one is absent)."""
    if "librosa" in sys.modules:
        return
    filters = types.ModuleType("librosa.filters")
    filters.mel = mel
    top = types.ModuleType("librosa")
    top.filters = filters
    top.__path__ = []   # behaves like a package, so `import librosa.filters` works
    sys.modules["librosa"] = top
    sys.modules["librosa.filters"] = filters
