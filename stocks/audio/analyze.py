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

# 5) downbeat: harmonic change between the bar before and the bar after each
#    beat (40..1000 Hz). Chords change on bar lines, so phase 0 of the bar wins.
S_lin = stft_mag(mono, 8192, sr // 100)
f = np.fft.rfftfreq(8192, 1 / sr)
band = (f > 40) & (f < 1000)
spec_t = (np.arange(S_lin.shape[0]) * (sr // 100) + 4096) / sr
grid = t0 + period * np.arange(int((t_frames[-1] - t0) / period) - 1)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


nov = []
for tb in grid:
    pre = (spec_t > tb - 4 * period) & (spec_t < tb - 0.03)
    post = (spec_t > tb + 0.03) & (spec_t < tb + 4 * period)
    u = unit(S_lin[pre][:, band].mean(0)) if pre.any() else None
    v = unit(S_lin[post][:, band].mean(0)) if post.any() else None
    nov.append(np.nan if u is None or v is None else 1 - float(u @ v))
nov = np.array(nov)
# Low-band (30..62 Hz, kick only) accent: the bar's first kick is played harder.
low = (f > 30) & (f < 62)
low_e = S_lin[:, low].sum(1)
acc = np.array([low_e[(spec_t > tb) & (spec_t < tb + 0.12)].mean() for tb in grid])
z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
score = z(acc) + 0.5 * np.nan_to_num(z(nov))
harm_scores = [float(np.nanmean(nov[p::4])) for p in range(4)]
phase_scores = [float(np.mean(score[p::4])) for p in range(4)]
db_phase = int(np.argmax(phase_scores))
downbeats = grid[db_phase::4]

# 6) which downbeat is bar 1 of the loop: the one with the biggest impact
#    (the band stops for half a beat, then hits). Broadband energy just after vs
#    just before. Take its second occurrence, so the pre-roll before the cut is
#    the loop itself.
hf = (f > 60) & (f < 12000)
hf_e = S_lin[:, hf].sum(1)
impact = []
for tb in downbeats:
    a = hf_e[(spec_t > tb) & (spec_t < tb + 0.25)].mean()
    m = (spec_t > tb - 0.2) & (spec_t < tb - 0.02)
    b = hf_e[m].mean() if m.any() else np.nan
    impact.append(a / (b + 1e-9))
impact = np.nan_to_num(np.array(impact))  # no pre-roll at the file start -> 0
# Score each bar position of the LOOP_BARS cycle by its median impact over all
# passes, so a one-off (e.g. the count-in -> band entrance) can't win.
bar_pos_score = [float(np.median(impact[p::LOOP_BARS])) for p in range(LOOP_BARS)]
best_pos = int(np.argmax(bar_pos_score))
cands = downbeats[best_pos::LOOP_BARS]
# Snap to the nearest whole BPM when the measurement is that close, so the loop
# is a whole number of samples and matches the video length exactly.
bpm_used = round(bpm) if abs(bpm - round(bpm)) < 0.01 else bpm
bar_len = 4 * 60 / bpm_used
start = cands[1] if len(cands) > 1 else cands[0]
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
    "harmonic_change_scores": [round(s, 4) for s in harm_scores],
    "downbeat_phase": db_phase,
    "impact_downbeats_s": [round(float(c), 3) for c in cands],
    "bar_position_median_impact": [round(v, 2) for v in bar_pos_score],
    "impact_top": sorted([(round(float(i), 2), round(float(d), 3)) for i, d in zip(impact, downbeats)], reverse=True)[:6],
    "loop_start_s": round(float(start), 4),
    "loop_len_s": round(float(loop_len), 4),
    "loop_seam_maxdiff": round(float(seam), 5),
}
(HERE / "grid.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
