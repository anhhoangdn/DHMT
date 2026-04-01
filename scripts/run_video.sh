#!/usr/bin/env bash
# Chạy full pipeline cho video
set -e

INPUT="${INPUT:-data/samples/test.mp4}"
OUTPUT="${OUTPUT:-outputs}"
DECA_ROOT="${DECA_ROOT:-external/DECA}"
DEVICE="${DEVICE:-cuda}"

echo "🎥  Chạy pipeline cho video: ${INPUT}"
python src/pipeline/run_pipeline.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --deca-root "${DECA_ROOT}" \
    --device "${DEVICE}"
