#!/usr/bin/env bash
# Chạy full pipeline cho ảnh tĩnh
set -e

INPUT="${INPUT:-data/samples/test.jpg}"
OUTPUT="${OUTPUT:-outputs}"
DECA_ROOT="${DECA_ROOT:-external/DECA}"
DEVICE="${DEVICE:-cuda}"

echo "🖼️  Chạy pipeline cho ảnh: ${INPUT}"
python src/pipeline/run_pipeline.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --deca-root "${DECA_ROOT}" \
    --device "${DEVICE}"
