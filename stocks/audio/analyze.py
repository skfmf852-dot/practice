"""Measure the beat grid of song_full.wav with numpy and cut a 7-bar loop
starting on a downbeat.

Writes audio/song_loop.wav and audio/grid.json.
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LOOP_BARS = 8


def read_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
        return x.reshape(-1, w.getnchannels()).T, sr


def stft_mag(x, n_fft, hop):
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1))


x, sr = read_wav(HERE / "song_full.wav")
mono = x.mean(axis=0)

# 1) onset envelope: positive spectral flux on a log spectrum, 2 ms hop
HOP = sr // 1000
NFFT = 1024
S = np.log1p(100 * stft_mag(mono, NFFT, HOP))
flux = np.maximum(np.diff(S, axis=0), 0).sum(axis=1)
flux = np.concatenate([[0], flux])
fps = sr / HOP
# Flux peaks when an onset sits ~3/4 into the Hann window, so time-stamp each
# frame there instead of at its centre (removes a ~N/4 early bias).
t_frames = (np.arange(len(flux)) * HOP + 0.75 * NFFT) / sr

# 2) tempo: autocorrelation of the onset envelope in the 70..180 BPM range
o = flux - flux.mean()
ac = np.fft.irfft(np.abs(np.fft.rfft(o, 2 * len(o))) ** 2)[: len(o)]
lags = np.arange(len(ac)) / fps
valid = (lags > 60 / 180) & (lags < 60 / 70)
k = np.argmax(np.where(valid, ac, -np.inf))
# parabolic refinement
a, b, c = ac[k - 1], ac[k], ac[k + 1]
k_ref = k + 0.5 * (a - c) / (a - 2 * b + c)
period0 = k_ref / fps

# 3) beat phase: comb-filter score over phases, then per-beat peak picking
n_beats = int((t_frames[-1] - 1.0) / period0)
best = None
for ph in np.arange(0, period0, 1 / fps):
    times = ph + period0 * np.arange(n_beats)
    score = np.interp(times, t_frames, flux).sum()
    if best is None or score > best[0]:
        best = (score, ph)
phase = best[1]
approx = phase + period0 * np.arange(n_beats)
beats = []
for tb in approx:
    m = (t_frames > tb - 0.03) & (t_frames < tb + 0.03)
    if m.sum() < 3:
        continue
    i = np.flatnonzero(m)[np.argmax(flux[m])]
    beats.append(t_frames[i])
beats = np.array(beats)
# keep beats that actually have an onset (drop the drumless intro bar)
strength = np.interp(beats, t_frames, flux)
beats = beats[strength > 0.25 * np.median(strength)]

# 4) linear fit of the grid -> precise period and offset
idx = np.round((beats - beats[0]) / period0)
A = np.vstack([idx, np.ones_like(idx)]).T
(period, t0), *_ = np.linalg.lstsq(A, beats, rcond=None)
resid_ms = (beats - (period * idx + t0)) * 1000
bpm = 60 / period

# 5) downbeat: harmonic change (bass + chords, 40..1000 Hz) right after each beat
S_lin = stft_mag(mono, 8192, sr // 100)
f = np.fft.rfftfreq(8192, 1 / sr)
band = (f > 40) & (f < 1000)
spec_t = (np.arange(S_lin.shape[0]) * (sr // 100) + 4096) / sr
grid = t0 + period * np.arange(int((t_frames[-1] - t0) / period) - 1)
nov = []
for tb in grid:
    pre = (spec_t > tb - 0.45) & (spec_t < tb - 0.05)
    post = (spec_t > tb + 0.05) & (spec_t < tb + 0.45)
    if pre.sum() == 0 or post.sum() == 0:
        nov.append(0)
        continue
    u = S_lin[pre][:, band].mean(0)
    v = S_lin[post][:, band].mean(0)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    nov.append(1 - float(u @ v))
nov = np.array(nov)
phase_scores = [nov[p::4].mean() for p in range(4)]
db_phase = int(np.argmax(phase_scores))
downbeats = grid[db_phase::4]

# 6) start on the downbeat that begins the second pass of the loop (so the cut
#    region's pre-roll is identical to what the loop itself provides)
# Snap to the nearest whole BPM when the measurement is that close, so the loop
# is a whole number of samples and matches the video length exactly.
bpm_used = round(bpm) if abs(bpm - round(bpm)) < 0.01 else bpm
bar_len = 4 * 60 / bpm_used
first_drum_bar = downbeats[np.argmin(np.abs(downbeats - beats[0]))]
start = first_drum_bar + LOOP_BARS * bar_len
start = downbeats[np.argmin(np.abs(downbeats - start))]
loop_len = LOOP_BARS * bar_len

i0 = int(round(start * sr))
n = int(round(loop_len * sr))
cut = x[:, i0 : i0 + n]
# What plays after the wrap (x[i0:]) must match what would naturally follow
# the cut's last sample (x[i0+n:]).
seam = np.abs(x[:, i0 + n : i0 + n + 4800] - x[:, i0 : i0 + 4800]).max()

pcm = (np.clip(cut.T, -1, 1) * 32767).astype("<i2")
with wave.open(str(HERE / "song_loop.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(pcm.tobytes())

report = {
    "bpm": round(bpm, 4),
    "bpm_used": bpm_used,
    "period_s": round(period, 6),
    "beats_detected": int(len(beats)),
    "grid_residual_ms": {"mean_abs": round(float(np.abs(resid_ms).mean()), 3), "max_abs": round(float(np.abs(resid_ms).max()), 3)},
    "downbeat_phase_scores": [round(s, 4) for s in phase_scores],
    "downbeat_phase": db_phase,
    "loop_start_s": round(float(start), 4),
    "loop_len_s": round(float(loop_len), 4),
    "loop_seam_maxdiff": round(float(seam), 5),
}
(HERE / "grid.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
