#!/usr/bin/env bash
# Blend 4 subframes per frame (motion blur) and mux the audio.
#   input: out/sub/s00000.png ... (60 fps x 4 subframes, in order)
#   tmix averages each group of 4; select keeps one output per group.
set -euo pipefail
cd "$(dirname "$0")/.."
ffmpeg -loglevel error -y -framerate 240 -i out/sub/s%05d.png -i audio/final.wav \
  -filter_complex "[0:v]tmix=frames=4:weights='1 1 1 1',select='eq(mod(n\,4)\,3)',setpts=N/(60*TB),format=yuv420p[v]" \
  -map "[v]" -map 1:a -r 60 -c:v libx264 -preset slow -crf 14 -tune animation -movflags +faststart \
  -c:a aac -b:a 256k -shortest out/stocks-up.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,r_frame_rate,width,height -show_entries format=duration,size -of default=nw=1 out/stocks-up.mp4
