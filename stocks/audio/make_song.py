"""Synthesize an original up-tempo big-band swing loop (128 BPM, 8 bars, D minor)
with numpy only. Genre reference: fast jazz with walking upright bass, bongos,
ride cymbal and brass-section hits. The melody, riffs and harmony here are
original; nothing is transcribed from an existing recording.

  bars 1-2  ensemble riff + stabs over walking bass and bongos
  bars 3-6  alto sax lead, brass pads behind it
  bars 7-8  shout chorus: chromatic stabs climbing on every beat, snare fill,
            band stops dead on the last half beat
  bar 1     big ensemble hit + crash on the downbeat (= the loop point)

Output: audio/song_full.wav (1 count-in bar + the loop 3 times)
"""
import wave
from pathlib import Path

import numpy as np

SR = 48000
BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SWING = 0.62          # off-beat 8th lands at 62% of the beat
LOOP_BARS = 8
INTRO_BARS = 1
PASSES = 3
TOTAL_BARS = INTRO_BARS + LOOP_BARS * PASSES
DUR = TOTAL_BARS * BAR + 3.0
N = int(DUR * SR)
rng = np.random.default_rng(0)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def e8(i):
    """Time of swung 8th i within a bar (i = 0..7)."""
    return (i // 2) * BEAT + (SWING * BEAT if i % 2 else 0.0)


def ta(sec):
    return np.arange(int(sec * SR)) / SR


def stem():
    return np.zeros((2, N))


def add(buf, t0, sig, pan=0.0, gain=1.0):
    i = int(round(t0 * SR))
    if i >= N or i < 0:
        return
    sig = sig[: N - i]
    l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
    r = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
    buf[0, i : i + len(sig)] += sig * gain * l
    buf[1, i : i + len(sig)] += sig * gain * r


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / SR)
    g = np.ones_like(f)
    if hi is not None:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
        g /= np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    return np.fft.irfft(X * g, n=x.shape[-1], axis=-1)


# ---------------------------------------------------------------- instruments
def upright(m, dur):
    """Plucked string (Karplus-Strong), computed one period-block at a time."""
    f = midi_hz(m)
    P = int(round(SR / f))
    n = int((dur + 0.15) * SR)
    y = np.zeros(n + P + 1)
    burst = rng.standard_normal(P)
    burst = np.convolve(burst, np.ones(6) / 6, mode="same")  # soft finger, not a pick
    y[:P] = burst
    d = 0.9965
    for i in range(P, n, P):
        a = y[i - P : i]
        b = np.concatenate(([y[i - P - 1] if i - P - 1 >= 0 else 0.0], y[i - P : i - 1]))
        y[i : i + P] = d * 0.5 * (a + b)
    y = y[:n]
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.08) * 0.6
    damp = np.clip((dur + 0.15 - t) / 0.08, 0, 1)
    return (y / (np.max(np.abs(y)) + 1e-9) + thump) * damp


def brass(m, dur, accent=1.0, bright=1.0, scoop=0.025, vib=0.0, sax=False):
    """Additive brass/sax: brightness swells on the attack ('blat'), pitch scoops up."""
    f0 = midi_hz(m)
    t = ta(dur + 0.08)
    att = 0.012 if not sax else 0.02
    amp = np.minimum(t / att, 1) * (0.8 + 0.2 * np.exp(-t / 0.08)) * np.clip((dur + 0.08 - t) / 0.06, 0, 1)
    B = bright * (0.55 + 0.45 * np.exp(-t / 0.09)) * np.minimum(t / 0.03, 1)  # brightness
    f = f0 * (1 - scoop * np.exp(-t / 0.035)) * (1 + vib * 0.006 * np.sin(2 * np.pi * 5.6 * t) * np.minimum(t / 0.25, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros_like(t)
    for kk in range(1, 26):
        if kk * f0 > 9000:
            break
        g = (1 / kk) * np.exp(-(kk - 1) * (1.15 - B) * (0.55 if not sax else 0.42))
        if sax and kk % 2 == 0:
            g *= 0.75
        out += g * np.sin(kk * ph)
    if sax:
        breath = fft_filter(rng.standard_normal(len(t))[None, :], lo=1500, hi=5000)[0]
        out += 0.05 * breath * amp
    return out * amp * accent


def ride(acc=1.0):
    t = ta(0.9)
    partials = [3120, 4410, 5230, 6170, 7400, 8650]
    x = sum(np.sin(2 * np.pi * p * t + i) * (1 / (i + 1)) for i, p in enumerate(partials))
    x = x * 0.25 + 0.35 * rng.standard_normal(len(t))
    return x * np.exp(-t / 0.32) * np.minimum(t / 0.001, 1) * acc


def chick():
    t = ta(0.05)
    return rng.standard_normal(len(t)) * np.exp(-t / 0.008)


def snare(acc=1.0):
    t = ta(0.2)
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.03)
    return (0.5 * body + rng.standard_normal(len(t)) * np.exp(-t / 0.06)) * acc


def kick(acc=1.0):
    t = ta(0.3)
    f = 50 + 70 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13) * np.minimum(t / 0.002, 1) * acc


def bongo(high=True, acc=1.0):
    t = ta(0.25)
    f0 = 410 if high else 290
    f = f0 * (1 + 0.25 * np.exp(-t / 0.01))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.07 if high else 0.1))
    slap = rng.standard_normal(len(t)) * np.exp(-t / 0.004) * 0.35
    return (tone + slap) * acc


def crash():
    t = ta(2.0)
    return rng.standard_normal(len(t)) * np.exp(-t / 0.6) * np.minimum(t / 0.002, 1)


# ---------------------------------------------------------------- the chart (original)
# Walking bass, quarter notes per bar. Bars 7-8 climb chromatically with the shout.
BASS = [
    [38, 41, 43, 44], [45, 43, 41, 42],      # Dm7 | Dm7
    [43, 46, 50, 49], [50, 48, 46, 45],      # Gm7 | Gm7
    [46, 50, 53, 44], [45, 49, 43, 40],      # Bbmaj7 | A7(b9)
    [39, 40, 41, 42], [43, 44, 45, None],    # Eb7 E7 F7 F#7 | G7 Ab7 A7 (stop)
]
# Four-voice close voicings for stabs (trombone, tenor, alto, trumpet).
V = {
    "Dm7": [57, 60, 62, 65], "Gm7": [58, 62, 65, 67], "Bb": [57, 62, 65, 69],
    "A7": [55, 58, 61, 64], "Eb7": [55, 58, 61, 63], "E7": [56, 59, 62, 64],
    "F7": [57, 60, 63, 65], "F#7": [58, 61, 64, 66], "G7": [59, 62, 65, 67],
    "Ab7": [60, 63, 66, 68], "A7b": [61, 64, 67, 69],
}
# Ensemble riff, bars 1-2 (bar, swung-8th index, midi, length in 8ths). Unison, octaves.
RIFF = [(0, 1, 62, 1), (0, 2, 65, 1), (0, 3, 67, 1), (0, 4, 68, 1), (0, 5, 69, 2),
        (0, 7, 72, 1), (1, 1, 69, 1), (1, 2, 67, 1)]
# Stabs: (bar, 8th, chord, length in 8ths, accent)
STABS = [(0, 0, "Dm7", 3, 1.25), (1, 3, "Dm7", 1, 1.0), (1, 5, "Dm7", 1, 1.0), (1, 7, "Dm7", 2, 1.1),
         (2, 0, "Gm7", 7, 0.32), (3, 0, "Gm7", 7, 0.32), (4, 0, "Bb", 7, 0.32), (5, 0, "A7", 6, 0.36),
         (6, 0, "Eb7", 1, 1.0), (6, 2, "E7", 1, 1.0), (6, 4, "F7", 1, 1.05), (6, 6, "F#7", 1, 1.05),
         (7, 0, "G7", 1, 1.1), (7, 2, "Ab7", 1, 1.15), (7, 4, "A7b", 2, 1.3)]
# Alto sax lead, bars 3-6.
SAX = [(2, 1, 74, 1), (2, 2, 77, 1), (2, 3, 79, 3), (2, 6, 77, 1), (2, 7, 74, 1),
       (3, 0, 72, 1), (3, 1, 70, 2), (3, 4, 67, 1), (3, 5, 70, 1), (3, 6, 72, 2),
       (4, 1, 74, 1), (4, 2, 77, 1), (4, 3, 81, 2), (4, 5, 79, 1), (4, 6, 77, 1), (4, 7, 76, 1),
       (5, 0, 73, 2), (5, 2, 76, 1), (5, 3, 79, 1), (5, 4, 82, 2), (5, 6, 81, 1), (5, 7, 79, 1)]
BONGO = [(0, True), (1, False), (2, True), (3, True), (4, False), (5, True), (6, False), (7, True)]

bassS, brassS, saxS, rideS, kitS, bongoS, fxS = (stem() for _ in range(7))


def dur8(n):
    return n * BEAT / 2 * 0.92


for bar in range(TOTAL_BARS):
    t_bar = bar * BAR
    if bar < INTRO_BARS:
        # count-in: ride + chick only
        rng = np.random.default_rng(900)
        for b in range(4):
            add(rideS, t_bar + b * BEAT, ride(0.8), pan=0.35)
            if b in (1, 3):
                add(kitS, t_bar + b * BEAT, chick(), pan=-0.3, gain=0.6)
        continue
    lb = (bar - INTRO_BARS) % LOOP_BARS
    rng = np.random.default_rng(1000 + lb)  # same seed per loop bar -> identical passes

    # walking bass (+ a swung ghost pickup into beat 3)
    for b, m in enumerate(BASS[lb]):
        if m is None:
            continue
        add(bassS, t_bar + b * BEAT, upright(m, BEAT * 0.95), gain=1.0 if b % 2 == 0 else 0.85)
    if lb in (0, 2, 4):
        add(bassS, t_bar + e8(3), upright(BASS[lb][1] + 12, BEAT * 0.3), gain=0.35)

    stop = lb == 7  # band stops on the last half beat of bar 8
    # ride: ding, ding-da, ding, ding-da ; chick on 2 & 4
    for b in range(4):
        if stop and b == 3:
            continue
        add(rideS, t_bar + b * BEAT, ride(0.9 if b % 2 == 0 else 1.0), pan=0.35)
        if b in (1, 3):
            add(rideS, t_bar + b * BEAT + SWING * BEAT, ride(0.6), pan=0.35)
            add(kitS, t_bar + b * BEAT, chick(), pan=-0.3, gain=0.7)
        add(kitS, t_bar + b * BEAT, kick(0.55 if b else 0.8))  # feathered kick
    # snare: ghost comping, then a fill in bar 8
    if lb < 6:
        for i in (3, 6):
            add(kitS, t_bar + e8(i), snare(0.25), pan=-0.1)
    elif lb == 6:
        for i in (1, 3, 5, 7):
            add(kitS, t_bar + e8(i), snare(0.45), pan=-0.1)
    else:
        for j in range(12):  # 8th-note triplets climbing for 3 beats
            add(kitS, t_bar + j * BEAT / 3, snare(0.35 + 0.65 * j / 12), pan=-0.1)
    # bongos: driving 8ths
    if lb < 6:
        for i, hi in BONGO:
            add(bongoS, t_bar + e8(i), bongo(hi, 0.9 if i % 2 == 0 else 0.7), pan=-0.45 if hi else -0.6)

    # ensemble riff (trumpet + alto + tenor in octaves)
    for rb, i, m, ln in RIFF:
        if rb == (lb if lb < 2 else -1):
            for o, pan, g in ((0, 0.25, 0.8), (12, -0.15, 0.55), (-12, 0.4, 0.5)):
                add(brassS, t_bar + e8(i), brass(m + o, dur8(ln), accent=1.1 if i % 2 else 1.0, vib=0.3), pan=pan, gain=g)
    # stabs / pads
    for sb, i, chord, ln, acc in STABS:
        if sb != lb:
            continue
        long_pad = acc < 0.5
        for v, m in enumerate(V[chord]):
            add(brassS, t_bar + e8(i), brass(m + (12 if v == 3 and not long_pad else 0), dur8(ln), accent=acc,
                                             bright=0.55 if long_pad else 1.0, scoop=0.0 if long_pad else 0.03),
                pan=[-0.5, -0.2, 0.2, 0.5][v], gain=0.5)
    # sax lead
    for sb, i, m, ln in SAX:
        if sb == lb:
            add(saxS, t_bar + e8(i), brass(m, dur8(ln), accent=1.0, scoop=0.03, vib=1.0 if ln > 1 else 0.3, sax=True), pan=0.1)
    # downbeat of the loop: crash
    if lb == 0:
        add(fxS, t_bar, crash(), gain=0.45, pan=0.2)

# ---------------------------------------------------------------- mix
bassS = fft_filter(bassS, lo=35, hi=1800)
rideS = fft_filter(rideS, lo=2500)
kitS = fft_filter(kitS, lo=40)
bongoS = fft_filter(bongoS, lo=120, hi=6000)
brassS = fft_filter(brassS, lo=120, hi=7000)
saxS = fft_filter(saxS, lo=150, hi=7000)
fxS = fft_filter(fxS, lo=3000)


def reverb(x, secs=1.3):
    r = np.random.default_rng(42)
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    L = x.shape[1] + n
    nfft = 1 << (L - 1).bit_length()
    for ch in range(2):
        ir = fft_filter((r.standard_normal(n) * np.exp(-t / (secs / 6.5)))[None, :], lo=300, hi=6000)[0]
        out[ch] = np.fft.irfft(np.fft.rfft(x[ch], nfft) * np.fft.rfft(ir, nfft), nfft)[: x.shape[1]]
    return out / (np.max(np.abs(out)) + 1e-9)


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


mix = (
    0.55 * norm(bassS)
    + 0.62 * norm(brassS)
    + 0.40 * norm(saxS)
    + 0.16 * norm(rideS)
    + 0.40 * norm(kitS)
    + 0.26 * norm(bongoS)
    + 0.30 * norm(fxS)
)
mix += 0.12 * reverb(norm(brassS) + 0.8 * norm(saxS) + 0.3 * norm(bongoS) + 0.3 * norm(kitS))
mix = np.tanh(1.2 * mix) / np.tanh(1.2)
mix = mix / np.max(np.abs(mix)) * 0.89

pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
out = Path(__file__).with_name("song_full.wav")
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"wrote {out} ({DUR:.2f} s, {TOTAL_BARS} bars @ {BPM} BPM, swing {SWING})")
