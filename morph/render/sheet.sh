#!/usr/bin/env bash
# Contact sheet: 7 columns x 4 rows (one row per... 28 images), each labelled with its time.
set -euo pipefail
dir=$1; outfile=$2
args=(); i=0
for f in $(ls "$dir"/t*.png | sort); do
  t=$(basename "$f" .png | sed 's/^t0*//; s/^\./0./')
  args+=(-i "$f")
done
n=$(ls "$dir"/t*.png | wc -l)
filter=""
for ((k=0;k<n;k++)); do
  t=$(ls "$dir"/t*.png | sort | sed -n "$((k+1))p" | xargs basename | sed 's/^t//; s/.png$//')
  filter+="[$k:v]scale=360:360,drawtext=text='$((k/4+1)).$((k%4+1))  ${t}s':x=10:y=10:fontsize=18:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw=4[v$k];"
done
ins=""; for ((k=0;k<n;k++)); do ins+="[v$k]"; done
ffmpeg -loglevel error -y "${args[@]}" -filter_complex "${filter}${ins}xstack=inputs=$n:grid=7x4:fill=white" "$outfile"
