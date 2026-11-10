"""Echo cancellation: remove the loudspeakers' sound from the microphone, using the
system-audio track as a reference.

The system track (BlackHole) is a digital copy of what the loudspeakers play, i.e. of the
other participants. The mic hears that sound too, after it crossed the room. What is left
of the mic once the loudspeaker part is subtracted is the user's voice (plus room noise), so
"is there anything left?" answers "is the user talking?" far more reliably than comparing
loudness, and the left-over audio gives a clean signal for the voiceprint.

Two steps:

1. Delay. The system track often arrives seconds late (Multi-Output Device), far more than an
   adaptive filter could model. It is measured with GCC-PHAT on the waveforms and removed.
2. Echo path. For every frequency bin, how the loudspeaker sound shows up in the mic is a few
   complex gains over neighbouring STFT frames (loudspeaker, room, mic and the reverb tail).
   They are fitted by least squares over the whole recording, then refitted with the frames
   where the user talks down-weighted so their voice does not leak into the estimate.

Run once on the whole recording after it stops. Limits: sound that the loudspeakers distort
non-linearly is only partly removed, and the delay is assumed constant over the recording."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft

SR = 16000

MAX_LAG_SEC = 4.5          # how far the system track may be out of step with the mic
LAG_CHUNK_SEC = 8.0
LAG_CHUNKS = 14            # how many stretches are used to measure the delay
LAG_BAND = (100.0, 4000.0)  # speech band used for GCC-PHAT
MIN_SHARPNESS = 6.0        # a correlation peak must stand out this much from the rest
MIN_GOOD_CHUNKS = 3
LAG_TOLERANCE_SEC = 0.05   # stretches must agree on the delay to this precision

NPERSEG = 4096             # 256 ms: long enough to hold the loudspeaker-to-mic path and its reverb tail
HOP = 1024
OFFSETS = (-1, 0, 1, 2)    # frames of the reference used per frame of the mic (slack either side, longer reverb)
CHUNK_SEC = 30.0
MARGIN_SEC = 1.0
ACTIVE_RMS = 2e-3          # the reference counts as playing above this loudness
MIN_ACTIVE_FRAMES = 200
RIDGE = 1e-3
_SPEC = float(np.sum(np.hanning(NPERSEG + 1)[:-1] ** 2) / np.sum(np.hanning(NPERSEG + 1)[:-1]) ** 2)


@dataclass
class EchoResult:
    residual: np.ndarray        # the mic with the loudspeaker sound removed (the mic itself if not ok)
    lag_sec: float | None       # how late the system track is
    erle_db: float | None       # how much quieter the mic got while the loudspeakers played
    ok: bool


# ---------- delay ----------

def _gcc_phat(mic_chunk: np.ndarray, ref_seg: np.ndarray, max_lag: int, sr: int) -> tuple[int, float]:
    """Delay (samples) at which ref_seg[t + lag] best matches mic_chunk[t], and how clear the peak is.
    ref_seg must start max_lag samples before the chunk and extend max_lag samples after it."""
    n = 1 << int(np.ceil(np.log2(len(mic_chunk) + len(ref_seg))))
    M = np.fft.rfft(mic_chunk, n)
    R = np.fft.rfft(ref_seg, n)
    G = R * np.conj(M)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    G[(freqs < LAG_BAND[0]) | (freqs > LAG_BAND[1])] = 0
    G /= np.abs(G) + 1e-12
    cc = np.fft.irfft(G, n)[: 2 * max_lag + 1]
    k = int(np.argmax(cc))
    return k - max_lag, float(cc[k] / (np.median(np.abs(cc)) + 1e-12))


def estimate_lag(mic: np.ndarray, ref: np.ndarray, sr: int = SR) -> int | None:
    """Samples by which the reference runs behind the mic (negative: ahead), None if the two
    recordings do not share enough sound to tell."""
    n = min(len(mic), len(ref))
    max_lag, size = int(MAX_LAG_SEC * sr), int(LAG_CHUNK_SEC * sr)
    if n < size * 2:
        return None
    starts = np.linspace(0, n - size, num=min(LAG_CHUNKS * 3, max(1, (n - size) // size)), dtype=int)
    loud = [(min(float(np.sqrt(np.mean(ref[a:a + size] ** 2))), float(np.sqrt(np.mean(mic[a:a + size] ** 2)))), a)
            for a in starts]
    picks = [a for level, a in sorted(loud, reverse=True)[:LAG_CHUNKS] if level > 1e-3]
    found = []
    for a in picks:
        lo = a - max_lag
        seg = np.zeros(size + 2 * max_lag, dtype=np.float32)
        src = ref[max(0, lo): min(n, a + size + max_lag)]
        seg[max(0, lo) - lo: max(0, lo) - lo + len(src)] = src
        lag, sharp = _gcc_phat(mic[a:a + size], seg, max_lag, sr)
        if sharp >= MIN_SHARPNESS:
            found.append(lag)
    if len(found) < MIN_GOOD_CHUNKS:
        return None
    mid = int(np.median(found))
    agree = [x for x in found if abs(x - mid) <= LAG_TOLERANCE_SEC * sr]
    return int(np.median(agree)) if len(agree) >= MIN_GOOD_CHUNKS else None


def advance(x: np.ndarray, samples: int) -> np.ndarray:
    """out[t] = x[t + samples] (zero-filled): moves a late recording back to where it belongs."""
    out = np.zeros_like(x)
    if samples > 0:
        out[: len(x) - samples] = x[samples:]
    elif samples < 0:
        out[-samples:] = x[: len(x) + samples]
    else:
        out[:] = x
    return out


# ---------- echo path ----------

def _shift_frames(a: np.ndarray, o: int) -> np.ndarray:
    """out[..., t] = a[..., t - o], zero where that frame does not exist."""
    out = np.zeros_like(a)
    if o > 0:
        out[..., o:] = a[..., :-o]
    elif o < 0:
        out[..., :o] = a[..., -o:]
    else:
        out[:] = a
    return out


def _spectra(mic: np.ndarray, ref: np.ndarray, a: int, b: int):
    """STFTs of mic and of the stacked reference taps for samples [a, b)."""
    _, _, D = stft(mic[a:b], fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    _, _, R = stft(ref[a:b], fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    X = np.stack([_shift_frames(R, o) for o in OFFSETS])           # (taps, bins, frames)
    return D.astype(np.complex128), X.astype(np.complex128), np.mean(np.abs(R) ** 2, axis=0)


def _chunks(n: int):
    """(start, end, kept_start, kept_end): a chunk with margins and the part of it that is kept."""
    step, margin = int(CHUNK_SEC * SR), int(MARGIN_SEC * SR)
    for s in range(0, n, step):
        e = min(n, s + step)
        yield max(0, s - margin), min(n, e + margin), s, e


def _solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    k = A.shape[-1]
    ridge = RIDGE * np.trace(A, axis1=1, axis2=2).real[:, None, None] / k + 1e-12
    return np.linalg.solve(A + ridge * np.eye(k), B[..., None])[..., 0]      # (bins, taps)


def _frame_ratio(D: np.ndarray, X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per frame: energy left after subtracting the echo / energy the mic heard."""
    E = D - np.einsum("fk,kft->ft", h, X)
    return (np.abs(E) ** 2).sum(0) / ((np.abs(D) ** 2).sum(0) + 1e-18)


def _fit(mic, ref, prev: np.ndarray | None = None, typical: float = 0.5) -> np.ndarray:
    """Least-squares echo path. With a previous fit, frames where the mic is far louder than the
    echo explains (the user talking) are down-weighted so their voice does not leak into the estimate."""
    A = B = None
    for a, b, _, _ in _chunks(len(mic)):
        D, X, P = _spectra(mic, ref, a, b)
        w = (P > _SPEC * ACTIVE_RMS ** 2).astype(float)
        if prev is not None:
            w = w * np.clip(2 * typical / (_frame_ratio(D, X, prev) + 1e-12), 0, 1) ** 2
        a_ = np.einsum("ift,jft,t->fij", X.conj(), X, w)
        b_ = np.einsum("ift,ft,t->fi", X.conj(), D, w)
        A = a_ if A is None else A + a_
        B = b_ if B is None else B + b_
    return _solve(A, B)


def _typical_ratio(mic, ref, h) -> float:
    """Median of (what is left) / (what the mic heard) over frames where the reference plays."""
    out = []
    for a, b, _, _ in _chunks(len(mic)):
        D, X, P = _spectra(mic, ref, a, b)
        out.append(_frame_ratio(D, X, h)[P > _SPEC * ACTIVE_RMS ** 2])
    out = np.concatenate(out) if out else np.zeros(0)
    return float(np.median(out)) if len(out) else 0.5


def cancel(mic: np.ndarray, ref: np.ndarray) -> EchoResult:
    """The mic with the loudspeaker sound removed. Falls back to the mic itself (ok=False) when
    the reference is missing, never plays, or cannot be lined up with the mic."""
    n = min(len(mic), len(ref))
    mic, ref = np.asarray(mic[:n], dtype=np.float32), np.asarray(ref[:n], dtype=np.float32)
    lag = estimate_lag(mic, ref)
    if lag is None:
        return EchoResult(mic, None, None, False)
    ref = advance(ref, lag)
    pad = NPERSEG
    mic_p, ref_p = np.pad(mic, (pad, pad)), np.pad(ref, (pad, pad))

    h = _fit(mic_p, ref_p)                                        # first fit: every frame where the reference plays
    h = _fit(mic_p, ref_p, h, _typical_ratio(mic_p, ref_p, h))    # second fit: user's talk down-weighted

    out = np.zeros(len(mic_p), dtype=np.float32)
    before = after = 0.0
    frames = 0
    for a, b, ka, kb in _chunks(len(mic_p)):
        D, X, P = _spectra(mic_p, ref_p, a, b)
        E = D - np.einsum("fk,kft->ft", h, X)
        with warnings.catch_warnings():      # the chunk's own edges are not reconstructable; they are discarded
            warnings.simplefilter("ignore")
            _, y = istft(E, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=False)
        lo = ka - a
        seg = y[lo: lo + (kb - ka)]
        out[ka: ka + len(seg)] = seg
        act = P > _SPEC * ACTIVE_RMS ** 2
        before += float((np.abs(D[:, act]) ** 2).sum())
        after += float((np.abs(E[:, act]) ** 2).sum())
        frames += int(act.sum())
    if frames < MIN_ACTIVE_FRAMES:
        return EchoResult(mic, lag / SR, None, False)
    return EchoResult(out[pad: pad + n], lag / SR, 10 * float(np.log10((before + 1e-18) / (after + 1e-18))), True)
