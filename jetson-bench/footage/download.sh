#!/usr/bin/env bash
# Downloads the three Pexels test clips into footage/raw (Pexels license, see README.md). The bench cuts frames itself with ffmpeg.
set -e
cd "$(dirname "$0")" && mkdir -p raw && cd raw
curl -sSL -o pexels-32243653-forklift-operator.mp4      https://videos.pexels.com/video-files/32243653/13751459_2560_1440_25fps.mp4
curl -sSL -o pexels-15170997-warehouse-team.mp4         https://videos.pexels.com/video-files/15170997/15170997-hd_1080_1920_30fps.mp4
curl -sSL -o pexels-6079421-forklift-busy-warehouse.mp4 https://videos.pexels.com/video-files/6079421/6079421-uhd_2560_1440_24fps.mp4
ls -la
