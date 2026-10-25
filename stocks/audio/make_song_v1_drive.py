"""(v1, replaced by make_song.py) Synthesize an urgent, driving 128 BPM, 8-bar loop with numpy only.

Shape of the loop (A minor):
  bars 1-6  drive: four-on-the-floor kick, rolling 16th bass, arp, pump
  bar 7     build: snare 8ths, arp up an octave, riser starts
  bar 8     build: snare 16ths climbing in pitch, riser peaks, kick drops
            out for the last half beat
  bar 1     drop: crash + sub boom on the downbeat (which is also the loop
            point, so the build always resolves into the start)

The song is 1 intro bar + the loop 3 times; we later cut the 2nd pass so the
material spilling over the seam is exactly what the loop itself produces.
Output: audio/song_full.wav
"""
import wave
from pathlib import Path

import numpy as np

SR = 48000
BPM = 128.0
BEAT = 60.0 / BPM
S16 = BEAT / 4
BAR = 4 * BEAT
LOOP_BARS = 8
INTRO_BARS = 1
PASSES = 3
TOTAL_BARS = INTRO_BARS + LOOP_BARS * PASSES
DUR = TOTAL_BARS * BAR + 3.0
N = int(DUR * SR)
rng = np.random.default_rng(0)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def stem():
    return np.zeros((2, N))


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / SR)
    g = np.ones_like(f)
    if hi is not None:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
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


def ta(sec):
    return np.arange(int(sec * SR)) / SR


# ---------------------------------------------------------------- instruments
def kick(accent=1.0):
    t = ta(0.38)
    f = 44 + 140 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.17) * np.minimum(t / 0.0012, 1)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.0018) * 0.3
    return (body + click) * accent


def clap():
    t = ta(0.28)
    e = sum(np.where(t >= o, np.exp(-np.maximum(t - o, 0) / 0.005), 0) for o in (0.0, 0.008, 0.016))
    tail = np.where(t >= 0.016, np.exp(-np.maximum(t - 0.016, 0) / 0.08), 0) * 0.6
    return rng.standard_normal(len(t)) * (e + tail)


def snare(pitch=1.0):
    t = ta(0.18)
    body = np.sin(2 * np.pi * 190 * pitch * t) * np.exp(-t / 0.03)
    noise = rng.standard_normal(len(t)) * np.exp(-t / 0.05)
    return 0.6 * body + noise


def hat(open_=False):
    t = ta(0.22 if open_ else 0.05)
    return rng.standard_normal(len(t)) * np.exp(-t / (0.08 if open_ else 0.014))


def crash():
    t = ta(1.8)
    return rng.standard_normal(len(t)) * np.exp(-t / 0.55) * np.minimum(t / 0.002, 1)


def boom():
    t = ta(1.2)
    f = 38 + 50 * np.exp(-t / 0.08)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45) * np.minimum(t / 0.003, 1)


def saw_additive(freq, n, max_f=6000.0, tilt=1.15):
    t = np.arange(n) / SR
    out = np.zeros(n)
    k = 1
    while k * freq < max_f and k < 48:
        out += np.sin(2 * np.pi * k * freq * t + rng.uniform(0, 2 * np.pi)) / k**tilt
        k += 1
    return out


def env(n, a, d, s, rel_at, r):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    return e * np.clip(1 - (t - rel_at) / r, 0, 1)


def pad_note(m, dur):
    n = int((dur + 0.3) * SR)
    f = midi_hz(m)
    l = saw_additive(f * 2 ** (-8 / 1200), n) + saw_additive(f * 2 ** (5 / 1200), n)
    r = saw_additive(f * 2 ** (8 / 1200), n) + saw_additive(f * 2 ** (-4 / 1200), n)
    e = env(n, 0.03, 0.4, 0.7, dur, 0.25)
    return l * e, r * e


def bass_note(m, dur):
    t = ta(dur + 0.03)
    f = midi_hz(m)
    sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.18 * np.sin(6 * np.pi * f * t) + 0.08 * np.sin(8 * np.pi * f * t)
    return sig * np.minimum(t / 0.003, 1) * np.exp(-t / 0.12) * np.clip((dur + 0.03 - t) / 0.02, 0, 1)


def pluck(m):
    t = ta(0.3)
    f = midi_hz(m)
    sig = np.sign(np.sin(2 * np.pi * f * t)) * 0.35 + np.sin(2 * np.pi * f * t)
    return sig * np.minimum(t / 0.002, 1) * np.exp(-t / 0.075)


def riser(sec):
    """Pitch + noise sweep that climbs over `sec` and stops dead at the end."""
    t = ta(sec)
    u = t / sec
    f = 180 * (16 ** u)  # 180 Hz -> 2.9 kHz
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5 + np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR) * 0.25
    noise = rng.standard_normal(len(t))
    return (tone + noise * (0.15 + 0.85 * u)) * (u**1.6)


# ---------------------------------------------------------------- arrangement
CHORDS = [  # (pad voicing, bass root)
    ([52, 57, 60, 64], 33),  # Am
    ([53, 57, 60, 65], 29),  # F
    ([52, 55, 60, 64], 36),  # C
    ([50, 55, 59, 62], 31),  # G
    ([52, 57, 60, 64], 33),  # Am
    ([53, 57, 60, 65], 29),  # F
    ([50, 53, 57, 62], 38),  # Dm
    ([52, 56, 59, 64], 40),  # E (dominant -> tension into the loop)
]
ARP = [0, 1, 2, 3, 2, 3, 1, 2]

kickS, clapS, snS, hatS, bassS, padS, arpS, fxS = (stem() for _ in range(8))
kick_times = []

for bar in range(TOTAL_BARS):
    t_bar = bar * BAR
    intro = bar < INTRO_BARS
    lb = (bar - INTRO_BARS) % LOOP_BARS
    rng = np.random.default_rng(500 if intro else 600 + lb)  # identical passes
    voicing, root = CHORDS[7 if intro else lb]

    # pad: off-beat stabs in the drive, sustained chord in the build
    if intro or lb >= 6:
        for m in voicing:
            l, r = pad_note(m, BAR - 0.04)
            add(padS, t_bar, l, pan=-0.6, gain=0.8)
            add(padS, t_bar, r, pan=0.6, gain=0.8)
    else:
        for beat in range(4):
            for m in voicing:
                l, r = pad_note(m, BEAT * 0.42)
                add(padS, t_bar + beat * BEAT + BEAT / 2, l, pan=-0.6)
                add(padS, t_bar + beat * BEAT + BEAT / 2, r, pan=0.6)

    # arp, an octave up in the build
    up = 12 + (12 if lb >= 6 and not intro else 0)
    for k in range(16):
        add(arpS, t_bar + k * S16, pluck(voicing[ARP[k % 8]] + up), pan=-0.3 if k % 2 else 0.3, gain=0.9 if k % 4 == 0 else 0.6)

    if intro:
        add(fxS, t_bar, riser(BAR), gain=0.6)
        continue

    for beat in range(4):
        tb = t_bar + beat * BEAT
        last_half = lb == 7 and beat == 3
        if not last_half:
            add(kickS, tb, kick(1.0 if beat == 0 else 0.9))
            kick_times.append(tb)
        if beat in (1, 3) and lb < 6:
            add(clapS, tb, clap())
        # hats: 16ths, open hat on the off-beat
        for s in range(4):
            if s == 2:
                add(hatS, tb + s * S16, hat(open_=True), pan=0.2, gain=0.8)
            else:
                add(hatS, tb + s * S16, hat(), pan=-0.25 if s % 2 else 0.25, gain=0.5 if s else 0.7)
        # rolling bass on the three 16ths after each kick
        if not last_half:
            for s, oct_ in ((1, 0), (2, 0), (3, 12)):
                add(bassS, tb + s * S16, bass_note(root + oct_, S16 * 0.85))

    # build: snare roll 8ths (bar 7) -> 16ths climbing in pitch and level (bar 8)
    if lb == 6:
        for k in range(8):
            add(snS, t_bar + k * BEAT / 2, snare(1.0), gain=0.35 + 0.35 * k / 8)
    if lb == 7:
        for k in range(16):
            add(snS, t_bar + k * S16, snare(1.0 + 0.9 * k / 16), gain=0.55 + 0.45 * k / 16)
    if lb == 6:
        add(fxS, t_bar, riser(2 * BAR - BEAT / 2), gain=1.0)
    # the drop: crash + boom on every loop downbeat
    if lb == 0:
        add(fxS, t_bar, crash(), gain=0.5)
        add(kickS, t_bar, boom(), gain=0.7)

# ---------------------------------------------------------------- processing
clapS = fft_filter(clapS, lo=900, hi=5000)
snS = fft_filter(snS, lo=150, hi=7000)
hatS = fft_filter(hatS, lo=7500, order=3)
padS = fft_filter(padS, hi=2600, order=2)
arpS = fft_filter(arpS, lo=200, hi=4500)
bassS = fft_filter(bassS, hi=700)
fxS = fft_filter(fxS, lo=120)


def delay(x, d, fb, taps=4):
    out = x.copy()
    sh = int(d * SR)
    g = 1.0
    for k in range(1, taps + 1):
        g *= fb
        y = np.zeros_like(x)
        y[:, sh * k :] = x[:, : -sh * k]
        out += g * (y[::-1] if k % 2 else y)
    return out


arpS = delay(arpS, BEAT * 0.75, 0.3)


def reverb(x, secs=1.6):
    r = np.random.default_rng(42)
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    L = x.shape[1] + n
    nfft = 1 << (L - 1).bit_length()
    for ch in range(2):
        ir = fft_filter((r.standard_normal(n) * np.exp(-t / (secs / 6.5)))[None, :], lo=250, hi=7000)[0]
        out[ch] = np.fft.irfft(np.fft.rfft(x[ch], nfft) * np.fft.rfft(ir, nfft), nfft)[: x.shape[1]]
    return out / (np.max(np.abs(out)) + 1e-9)


t_axis = np.arange(N) / SR
duck = np.ones(N)
for kt in kick_times:
    i = int(kt * SR)
    m = int(0.3 * SR)
    seg = t_axis[i : i + m] - kt
    duck[i : i + m] = np.minimum(duck[i : i + m], 1 - 0.6 * np.exp(-seg / 0.08) * np.minimum(seg / 0.003, 1))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


mix = (
    0.95 * norm(kickS)
    + 0.28 * norm(clapS)
    + 0.30 * norm(snS)
    + 0.12 * norm(hatS)
    + 0.50 * norm(bassS) * duck
    + 0.22 * norm(padS) * duck
    + 0.20 * norm(arpS) * duck
    + 0.20 * norm(fxS)
)
mix += 0.09 * reverb(0.7 * norm(padS) + norm(arpS) + 0.6 * norm(snS) + 0.4 * norm(clapS)) * duck
mix = np.tanh(1.25 * mix) / np.tanh(1.25)
mix = mix / np.max(np.abs(mix)) * 0.89

pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
out = Path(__file__).with_name("song_full_v1.wav")
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"wrote {out} ({DUR:.2f} s, {TOTAL_BARS} bars @ {BPM} BPM)")
