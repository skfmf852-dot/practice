"""Synthesize the UI sounds and mix them over the song loop.

Every sound is placed by its measured peak: we find the sample where the
sound's envelope peaks and shift it so that sample lands exactly on the event
time from window.SOUNDS (dumped to audio/sounds.json by render/render.js).
The mix wraps around the loop seam, so a sound near the end bleeds into the
start just like it would when the loop repeats.

Writes audio/final.wav and audio/sound_report.json.
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SR = 48000
rng = np.random.default_rng(3)


def t_axis(sec):
    return np.arange(int(sec * SR)) / SR


def hp_noise(n, cutoff):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 1 / np.sqrt(1 + (cutoff / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X, n)


def tone(f, decay, sec=0.25, attack=0.0008):
    t = t_axis(sec)
    return np.sin(2 * np.pi * f * t) * np.minimum(t / attack, 1) * np.exp(-t / decay)


def mixs(*parts):
    """Sum sounds of different lengths."""
    out = np.zeros(max(len(x) for x in parts))
    for x in parts:
        out[: len(x)] += x
    return out


def click(f=1900, body=0.006, noise=0.5):
    n = int(0.08 * SR)
    t = t_axis(0.08)
    return tone(f, body, 0.08) + noise * hp_noise(n, 3500) * np.exp(-t / 0.0025) * np.minimum(t / 0.0004, 1)


SOUNDS = {
    "click": lambda: mixs(click(1900, 0.006), 0.5 * tone(420, 0.012, 0.08)),
    "tab": lambda: 0.7 * click(2400, 0.005, 0.35),
    "tick": lambda: 0.35 * click(3200, 0.003, 0.2),
    "grab": lambda: mixs(0.8 * click(1100, 0.008, 0.3), 0.4 * tone(300, 0.015, 0.08)),
    "release": lambda: 0.7 * click(1500, 0.006, 0.3),
    "toggle": lambda: mixs(click(2100, 0.005, 0.6), 0.9 * tone(240, 0.02, 0.1)),
    "key": lambda: mixs(0.55 * click(rng.uniform(1500, 2100), 0.004, 0.9), 0.25 * tone(rng.uniform(500, 700), 0.01, 0.08)),
    "enter": lambda: mixs(0.8 * click(1200, 0.006, 0.9), 0.6 * tone(260, 0.03, 0.12)),
    "success": lambda: 0.6 * tone(1318.5, 0.12, 0.6) + 0.28 * np.concatenate([np.zeros(int(0.07 * SR)), tone(1975.5, 0.16, 0.6)])[: int(0.6 * SR)],
    "toast": lambda: 0.4 * tone(987.8, 0.1, 0.5) + 0.13 * np.concatenate([np.zeros(int(0.06 * SR)), tone(1318.5, 0.14, 0.5)])[: int(0.5 * SR)],
}


def peak_index(x):
    # envelope peak: abs value smoothed over 1 ms
    k = int(0.001 * SR)
    env = np.convolve(np.abs(x), np.ones(k) / k, mode="same")
    return int(np.argmax(env))


def read_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
        return x.reshape(-1, w.getnchannels()).T


song = read_wav(HERE / "song_loop.wav")
spec = json.loads((HERE / "sounds.json").read_text())
L = song.shape[1]
assert abs(L / SR - spec["T"]) < 1e-3, (L / SR, spec["T"])

ui = np.zeros(L)
report = []
for t_ev, name in spec["events"]:
    s = SOUNDS[name]()
    pk = peak_index(s)
    start = int(round(t_ev * SR)) - pk
    idx = (start + np.arange(len(s))) % L  # wrap across the loop seam
    np.add.at(ui, idx, s)
    report.append({"t": t_ev, "sound": name, "peak_ms_into_sound": round(pk / SR * 1000, 2), "placed_at_s": round((start + pk) / SR, 5)})

ui = ui / np.max(np.abs(ui)) * 0.42
mix = song * 0.86 + ui[None, :]
mix = np.tanh(1.1 * mix) / np.tanh(1.1)
mix = mix / np.max(np.abs(mix)) * 0.89

pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
with wave.open(str(HERE / "final.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
(HERE / "sound_report.json").write_text(json.dumps(report, indent=1))
err = max(abs(r["placed_at_s"] - r["t"]) for r in report)
print(f"{len(report)} UI sounds placed; max peak offset from event = {err * 1000:.3f} ms")
