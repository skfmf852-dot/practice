"""Synthesize a 120 BPM, 7-bar looping track with numpy only.

The song is: 1 intro bar + the 7-bar loop played 3 times. The second pass of
the loop is the one we cut, so reverb tails and pad releases that spill over
the loop seam are exactly what the loop would produce when it repeats.

Output: audio/song_full.wav (stereo, 48 kHz, 16-bit)
"""
import wave
from pathlib import Path

import numpy as np

SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
LOOP_BARS = 7
INTRO_BARS = 1
PASSES = 3
TOTAL_BARS = INTRO_BARS + LOOP_BARS * PASSES
DUR = TOTAL_BARS * BAR + 3.0  # tail room

rng = np.random.default_rng(7)
N = int(DUR * SR)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def stem():
    return np.zeros((2, N))


def fft_filter(x, lo=None, hi=None, order=2):
    """Zero-phase band shaping in the frequency domain (Butterworth-like magnitude)."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / SR)
    g = np.ones_like(f)
    if hi is not None:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
        with np.errstate(divide="ignore"):
            g /= np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    return np.fft.irfft(X * g, n=x.shape[-1], axis=-1)


def add(buf, t0, sig, pan=0.0, gain=1.0):
    i = int(round(t0 * SR))
    if i >= N:
        return
    sig = sig[: N - i]
    l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
    r = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
    buf[0, i : i + len(sig)] += sig * gain * l
    buf[1, i : i + len(sig)] += sig * gain * r


def env_adsr(n, a, d, s, r_start, r):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    rel = np.clip((t - r_start) / r, 0, None)
    return e * np.exp(-rel * 5) * (t < r_start + r)


# ---------------------------------------------------------------- instruments
def kick(accent=1.0):
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    f = 46 + 120 * np.exp(-t / 0.032)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.2) * np.minimum(t / 0.0015, 1)
    click = rng.standard_normal(n) * np.exp(-t / 0.002) * 0.25
    return (body + click) * accent


def clap():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    e = np.zeros(n)
    for k, off in enumerate((0.0, 0.009, 0.018)):
        tt = t - off
        e += np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.006), 0) * (0.8 if k < 2 else 1.0)
    tail = np.where(t >= 0.018, np.exp(-np.maximum(t - 0.018, 0) / 0.09), 0) * 0.55
    return noise * (e + tail)


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    return rng.standard_normal(n) * np.exp(-t / (0.09 if open_ else 0.018))


def saw_additive(freq, n, max_f=5000.0, tilt=1.25):
    t = np.arange(n) / SR
    out = np.zeros(n)
    k = 1
    while k * freq < max_f and k < 40:
        out += np.sin(2 * np.pi * k * freq * t + rng.uniform(0, 2 * np.pi)) / k**tilt
        k += 1
    return out


def pad_note(m, dur):
    n = int((dur + 0.6) * SR)
    f = midi_hz(m)
    left = saw_additive(f * 2 ** (-6 / 1200), n) + saw_additive(f * 2 ** (4 / 1200), n)
    right = saw_additive(f * 2 ** (6 / 1200), n) + saw_additive(f * 2 ** (-3 / 1200), n)
    e = env_adsr(n, 0.12, 0.6, 0.75, dur, 0.5)
    return left * e, right * e


def bass_note(m, dur):
    n = int((dur + 0.08) * SR)
    t = np.arange(n) / SR
    f = midi_hz(m)
    sig = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
    e = np.minimum(t / 0.004, 1) * np.exp(-t / 0.5) * np.clip((dur + 0.08 - t) / 0.06, 0, 1)
    return sig * e


def pluck(m):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = midi_hz(m)
    sig = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.05)
    return sig * np.minimum(t / 0.002, 1) * np.exp(-t / 0.11)


# ---------------------------------------------------------------- arrangement
# (pad voicing, bass root) per bar; bar 7 splits halfway.
CHORDS = [
    ([53, 57, 60, 64], 38),  # Dm9
    ([53, 57, 58, 62], 34),  # Bbmaj7
    ([52, 57, 60, 65], 41),  # Fmaj7
    ([52, 55, 57, 62], 36),  # C6/9
    ([53, 57, 60, 64], 38),  # Dm9
    ([53, 57, 58, 62], 34),  # Bbmaj7
    ("split", [([50, 53, 57, 58], 31), ([55, 57, 62, 64], 33)]),  # Gm9 -> A7sus4
]
BASS_8THS = [(0, 0.38, 0), (3, 0.2, 0), (4, 0.38, 0), (6, 0.2, 12), (7, 0.2, 0)]
ARP = [0, 1, 2, 3, 2, 1, 2, 3]

drums, bassS, padS, arpS, clapS, hatS = stem(), stem(), stem(), stem(), stem(), stem()
kick_times = []

for bar in range(TOTAL_BARS):
    t_bar = bar * BAR
    intro = bar < INTRO_BARS
    loop_bar = (bar - INTRO_BARS) % LOOP_BARS
    # same seed for the same bar of every pass, so all passes are sample-identical
    rng = np.random.default_rng(999 if intro else 1000 + loop_bar)
    spec = CHORDS[loop_bar if not intro else 0]
    halves = spec[1] if spec[0] == "split" else [spec, spec]

    for beat in range(4):
        tb = t_bar + beat * BEAT
        if not intro:
            add(drums, tb, kick(1.0 if beat == 0 else 0.86))
            kick_times.append(tb)
            if beat in (1, 3):
                add(clapS, tb, clap(), pan=0.05)
        # hats on every 8th off-beat, quiet 16ths
        add(hatS, tb + BEAT / 2, hat(open_=(beat == 3 and loop_bar % 2 == 1)), pan=0.25, gain=0.9)
        for s16 in (1, 3):
            add(hatS, tb + s16 * BEAT / 4, hat(), pan=-0.2, gain=0.35)

    for half in range(2):
        voicing, root = halves[half]
        th = t_bar + half * 2 * BEAT
        # pad: each half-bar retriggers only when the chord actually changes
        if half == 0 or spec[0] == "split":
            dur = (2 * BEAT if spec[0] == "split" else BAR) - 0.05
            for m in voicing:
                l, r = pad_note(m, dur)
                add(padS, th, l, pan=-0.6)
                add(padS, th, r, pan=0.6)
        # arp 16ths over the half bar
        for k in range(8):
            m = voicing[ARP[k]] + 12
            add(arpS, th + k * BEAT / 4, pluck(m), pan=-0.35 if k % 2 == 0 else 0.35, gain=0.8 if k % 4 == 0 else 0.55)

    if not intro:
        for idx, ln, oct_ in BASS_8THS:
            half = 0 if idx < 4 else 1
            root = halves[half][1]
            add(bassS, t_bar + idx * BEAT / 2, bass_note(root + oct_, ln))

# ---------------------------------------------------------------- processing
clapS = fft_filter(clapS, lo=900, hi=4200)
hatS = fft_filter(hatS, lo=7000, order=3)
padS = fft_filter(padS, hi=1800, order=2)
arpS = fft_filter(arpS, hi=5000)
bassS = fft_filter(bassS, hi=900)


def delay(x, d, fb, n_taps=5):
    out = x.copy()
    shift = int(d * SR)
    g = 1.0
    for k in range(1, n_taps + 1):
        g *= fb
        y = np.zeros_like(x)
        y[:, shift * k :] = x[:, : -shift * k]
        out += g * (y[::-1] if k % 2 else y)  # ping-pong
    return out


arpS = delay(arpS, 0.375, 0.38)


def reverb(x, secs=2.2):
    rng = np.random.default_rng(42)
    n = int(secs * SR)
    t = np.arange(n) / SR
    irs = []
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / (secs / 6.5))
        irs.append(fft_filter(ir[None, :], lo=200, hi=6000)[0])
    out = np.zeros_like(x)
    L = x.shape[1] + n
    nfft = 1 << (L - 1).bit_length()
    for ch in range(2):
        y = np.fft.irfft(np.fft.rfft(x[ch], nfft) * np.fft.rfft(irs[ch], nfft), nfft)[: x.shape[1]]
        out[ch] = y
    return out / np.max(np.abs(out) + 1e-9)


# sidechain pump keyed from the kicks
t_axis = np.arange(N) / SR
duck = np.ones(N)
for kt in kick_times:
    i = int(kt * SR)
    m = int(0.32 * SR)
    seg = t_axis[i : i + m] - kt
    duck[i : i + m] = np.minimum(duck[i : i + m], 1 - 0.55 * np.exp(-seg / 0.09) * np.minimum(seg / 0.004, 1))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


mix = (
    0.95 * norm(drums)
    + 0.30 * norm(clapS)
    + 0.13 * norm(hatS)
    + 0.48 * norm(bassS) * duck
    + 0.24 * norm(padS) * duck
    + 0.17 * norm(arpS) * duck
)
wet = reverb(0.6 * norm(padS) + norm(arpS) + 0.5 * norm(clapS)) * duck
mix = mix + 0.10 * wet
mix = np.tanh(1.15 * mix) / np.tanh(1.15)
mix = mix / np.max(np.abs(mix)) * 0.89  # about -1 dBFS


def write_wav(path, x):
    pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


out = Path(__file__).with_name("song_full.wav")
write_wav(out, mix)
print(f"wrote {out} ({DUR:.1f} s, {TOTAL_BARS} bars, intro {INTRO_BARS})")
