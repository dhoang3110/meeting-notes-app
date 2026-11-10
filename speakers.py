"""Tell the user's own speech ("Me") from the other participants' ("Others").

The recorder captures the microphone and the system audio (BlackHole) as separate tracks.
Whoever is talking shows up as energy on one of them: the user on the mic, everyone else
on the system track. Transcription still runs once on the mixed audio; here every
transcript segment is labelled by comparing the two tracks' loudness over its time span.

Without headphones the microphone also picks up the loudspeakers, so the mic is never
silent while others talk. That leakage ("bleed") is not a constant (it varies with
who talks and how loud), so the decision leans on what is robust:

* the mic is loud while the system track is silent: the user is talking ("Me");
* both are active: the mic is mostly the loudspeakers, so it only counts as "Me" when it is
  far louder than the usual leakage AND its loudness does not follow the system track's
  (leakage rises and falls with the system audio, the user's voice does not).
  Overlapping speech therefore tends to be labelled "Others".

The system track can arrive late: with a Multi-Output Device the loudspeakers play at once but
BlackHole's copy may lag by seconds. The mic then hears others before the system track does,
and "mic loud, system silent" would wrongly read as the user. The lag is measured for each
recording (cross-correlation of the two loudness curves) and removed before comparing.

When echo cancellation (aec.py) produced a residual, the loudness rules for "both active" are
replaced by something much stronger: whether anything is left of the mic after the loudspeaker
sound is subtracted. The residual also gates the voiceprint: a match counts only if the
residual shows a voice beyond the leftovers of the loudspeakers.

If the user enrolled a voiceprint (voiceprint.py), each segment also gets a voice score and
that decides instead: it recognises the voice itself, so loudspeaker leakage, a muted mic
or overlapping speech no longer matter. Segments too short to score fall back to the rules above."""

import numpy as np

SR = 16000
HOP_SEC = 0.1

ME = "Me"
OTHERS = "Others"

SYSTEM_DEVICE_HINTS = ("blackhole", "loopback", "soundflower", "system audio (built-in)")

MIN_MIC_RMS = 1e-3        # the mic must be at least this loud to count as speech
SYS_ACTIVE_RMS = 2e-3     # the system track is "playing" above this level
NOISE_PERCENTILE = 10     # mic noise floor = quietest tenth of the recording
STRONG_FRACTION = 0.5     # "clearly playing": at least this share of the system track's 75th-percentile level
MAX_NOISE_RMS = 3e-3      # a mic's own noise floor never counts as louder than this
BLEED_PERCENTILE = 40     # leakage = this percentile of mic/system while the system plays (a median would
                          # drift up if the user talks over others for half of it; a low one reads too low)
MIN_BLEED_FRAMES = 20     # fewer active frames than this (2 s) and the leakage is unknown
NOISE_MARGIN = 2.0        # mic must exceed this many times the noise floor
OVERLAP_RATIO = 3.0       # both active: mic must be this many times the usual leakage ...
OVERLAP_CORR_MAX = 0.5    # ... and correlate with the system track less than this
PAD_SEC = 0.4             # transcript timestamps are approximate: look this far around a segment
MAX_LAG_SEC = 0.8         # the loudspeaker sound reaches the mic a little later
MIN_CORR_FRAMES = 8       # shorter than this (0.8 s) is too little to judge a correlation
MAX_LAG_SEARCH_SEC = 4.0  # how far the system track may be out of step with the mic
MIN_LAG_FRAMES = 300      # need at least 30 s of audio to measure the lag
LAG_MIN_CORR = 0.25       # the best alignment must correlate at least this much ...
LAG_MIN_GAIN = 0.08       # ... and beat "no shift" by this much, else assume no lag
RES_MIN_RATIO = 0.2       # with a residual: it must be at least this share of the mic ...
RES_TYPICAL_MULT = 3.0    # ... and this many times what the residual is while only others talk
SILENT_SLACK = 0.10       # voice score may fall this far below the threshold when the system track is silent
BUSY_EXTRA = 0.10         # ... and must clear it by this much while others are audible (leakage of their
                          # voices on the mic can pass for a partial match)


def role_of(device_name: str) -> str:
    """'system' for a loopback device such as BlackHole, otherwise 'mic'."""
    name = (device_name or "").lower()
    return "system" if any(h in name for h in SYSTEM_DEVICE_HINTS) else "mic"


def frame_rms(audio: np.ndarray, sr: int = SR, hop_sec: float = HOP_SEC) -> np.ndarray:
    """Loudness (RMS) of every hop_sec of audio."""
    hop = int(sr * hop_sec)
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(audio[: n * hop], dtype=np.float32).reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1)).astype(np.float32)


class RunningEnvelope:
    """Loudness of a track that arrives in pieces (live recording). Keeps only one number
    per 0.1 s, plus the few samples that do not yet fill a whole frame."""

    def __init__(self, sr: int = SR, hop_sec: float = HOP_SEC):
        self._sr, self._hop_sec = sr, hop_sec
        self._rest = np.zeros(0, dtype=np.float32)
        self._frames = np.zeros(0, dtype=np.float32)

    def add(self, audio: np.ndarray) -> None:
        buf = np.concatenate([self._rest, np.asarray(audio, dtype=np.float32)])
        env = frame_rms(buf, self._sr, self._hop_sec)
        self._rest = buf[len(env) * int(self._sr * self._hop_sec):]
        if len(env):
            self._frames = np.concatenate([self._frames, env])

    def array(self) -> np.ndarray:
        return self._frames.copy()


def merge_envelopes(envs: list[np.ndarray]) -> np.ndarray:
    """Combine several tracks of the same role: the loudest one wins at each moment."""
    length = max((len(e) for e in envs), default=0)
    out = np.zeros(length, dtype=np.float32)
    for e in envs:
        out[: len(e)] = np.maximum(out[: len(e)], e)
    return out


def _noise_floor(mic: np.ndarray, sys: np.ndarray) -> float:
    """Mic level when nobody is talking. Measured while the system track is silent: with
    the loudspeakers on, the mic is never quiet, and a floor taken from all of it would
    swallow the leakage. Capped, since a real mic's noise is far below speech."""
    quiet = sys <= SYS_ACTIVE_RMS
    if int(quiet.sum()) >= MIN_BLEED_FRAMES:
        floor = float(np.percentile(mic[quiet], NOISE_PERCENTILE))
    else:
        floor = float(np.percentile(mic, 1)) if len(mic) else 0.0
    return min(floor, MAX_NOISE_RMS)


def _bleed(mic: np.ndarray, sys: np.ndarray, noise: float) -> float:
    """How much of the system track's loudness shows up on the mic when the user is quiet.
    Taken as a middling ratio over the frames where the system track is clearly playing:
    there the user is rarely talking too, and the per-frame ratio is noisy (the sound takes
    a moment to reach the mic), so a middling percentile beats a low one, which reads too low."""
    active = sys > SYS_ACTIVE_RMS
    if int(active.sum()) < MIN_BLEED_FRAMES:
        return 0.0
    strong = active & (sys >= STRONG_FRACTION * float(np.percentile(sys[active], 75)))
    if int(strong.sum()) < MIN_BLEED_FRAMES:
        strong = active
    ratio = np.clip(mic[strong] - noise, 0.0, None) / sys[strong]
    return float(np.percentile(ratio, BLEED_PERCENTILE))


def _window_rms(env: np.ndarray, start: float, end: float, hop_sec: float) -> float:
    i0 = max(0, int(start / hop_sec))
    i1 = max(i0 + 1, int(np.ceil(end / hop_sec)))
    chunk = env[i0:i1]
    return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0


def _follows(mic: np.ndarray, sys: np.ndarray, start: float, end: float, hop_sec: float) -> float | None:
    """How closely the mic's loudness tracks the system track's over the segment (best
    Pearson correlation over small delays). None when the segment is too short or flat."""
    i0 = max(0, int(start / hop_sec))
    i1 = int(np.ceil(end / hop_sec))
    if i1 - i0 < MIN_CORR_FRAMES or i1 > len(mic):
        return None
    a = mic[i0:i1]
    if a.std() < 1e-6:
        return None
    best = None
    for lag in range(-int(MAX_LAG_SEC / hop_sec), int(MAX_LAG_SEC / hop_sec) + 1):
        j0 = i0 - lag
        if j0 < 0 or j0 + (i1 - i0) > len(sys):
            continue
        b = sys[j0 : j0 + (i1 - i0)]
        if b.std() < 1e-6:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        best = c if best is None else max(best, c)
    return best


def estimate_lag(mic: np.ndarray, sys: np.ndarray, hop_sec: float = HOP_SEC) -> int:
    """Frames by which the system track runs behind the mic (negative: ahead), 0 if unclear.
    While others speak, the mic's loudness is a copy of the system track's; the shift that
    makes the two curves line up best is the lag."""
    n = min(len(mic), len(sys))
    if n < MIN_LAG_FRAMES:
        return 0
    m, s = mic[:n].astype(np.float64), sys[:n].astype(np.float64)
    reach = int(MAX_LAG_SEARCH_SEC / hop_sec)
    best_c, best_lag, c0 = -1.0, 0, 0.0
    for lag in range(-reach, reach + 1):
        a = m[max(0, -lag): n - max(0, lag)]
        b = s[max(0, lag): n - max(0, -lag)]
        if len(a) < MIN_LAG_FRAMES // 2 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if lag == 0:
            c0 = c
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag if best_c >= LAG_MIN_CORR and best_c - c0 >= LAG_MIN_GAIN else 0


def shift_earlier(env: np.ndarray, frames: int) -> np.ndarray:
    """out[t] = env[t + frames]: moves a late curve back to where it belongs (zero-filled)."""
    if frames == 0:
        return env
    out = np.zeros_like(env)
    if frames > 0:
        out[: len(env) - frames] = env[frames:]
    else:
        out[-frames:] = env[: len(env) + frames]
    return out


def label_segments(segments: list[dict], mic_env: np.ndarray, sys_env: np.ndarray,
                   hop_sec: float = HOP_SEC, voice: list[float | None] | None = None,
                   voice_threshold: float = 0.45, residual: np.ndarray | None = None) -> list[dict]:
    """Copy of segments with a "speaker" field: "Me" or "Others".
    voice: optional voiceprint similarity per segment (None where too short to score);
    it is also stored on each segment as "voice".
    residual: optional loudness curve of the mic after echo cancellation (same hop as mic_env)."""
    n = min(len(mic_env), len(sys_env))
    mic, sys = mic_env[:n], sys_env[:n]
    # The same speech sits in the mixed audio twice: the mic's copy (early) and the system track's
    # copy (late). A segment may come from either, so it is compared with the system track both
    # as recorded and with the lag removed.
    sys_raw = sys
    sys = shift_earlier(sys_raw, estimate_lag(mic, sys_raw, hop_sec))
    noise = _noise_floor(mic, sys) if n else 0.0
    bleed = _bleed(mic, sys, noise) if n else 0.0
    mic_floor = max(MIN_MIC_RMS, NOISE_MARGIN * noise)
    res, res_min = None, RES_MIN_RATIO
    if residual is not None and len(residual) >= n:
        res = residual[:n]
        busy = (np.maximum(sys_raw, sys) > SYS_ACTIVE_RMS) & (mic > mic_floor)
        if int(busy.sum()) >= MIN_BLEED_FRAMES:   # how much is left while only others talk
            res_min = max(RES_MIN_RATIO, RES_TYPICAL_MULT * float(np.median(res[busy] / mic[busy])))
    out = []
    for k, s in enumerate(segments):
        m = _window_rms(mic, s["start"], s["end"], hop_sec)
        near = max(_window_rms(sys_raw, s["start"] - PAD_SEC, s["end"] + PAD_SEC, hop_sec),
                   _window_rms(sys, s["start"] - PAD_SEC, s["end"] + PAD_SEC, hop_sec))
        score = voice[k] if voice is not None else None
        busy_now = near > SYS_ACTIVE_RMS
        left = None                        # does the mic hold a voice beyond the loudspeaker sound?
        if res is not None and busy_now and m > mic_floor:
            m_res = _window_rms(res, s["start"], s["end"], hop_sec)
            left = m_res > mic_floor and m_res / m > res_min
        if m <= mic_floor:
            me = False
        elif score is not None:
            # System track silent: the mic hears someone in the room, so the voice only has to be
            # roughly the user's. Others audible: it must clearly be, since their voices leaking
            # into the mic can score as a partial match, and the residual must show a voice too.
            me = score >= voice_threshold + (BUSY_EXTRA if busy_now else -SILENT_SLACK)
            if left is False:
                me = False
        elif not busy_now:
            me = True                      # the mic hears speech, the system track is silent
        elif left is not None:
            me = left
        else:
            y = max(_window_rms(sys_raw, s["start"], s["end"], hop_sec), _window_rms(sys, s["start"], s["end"], hop_sec))
            corr = _follows(mic, sys, s["start"], s["end"], hop_sec)
            me = (corr is not None and corr < OVERLAP_CORR_MAX
                  and (m - noise) > OVERLAP_RATIO * bleed * y)
        labelled = {**s, "speaker": ME if me else OTHERS}
        if score is not None:
            labelled["voice"] = round(float(score), 3)
        out.append(labelled)
    return out
