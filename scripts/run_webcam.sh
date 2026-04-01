#!/usr/bin/env bash
# Chạy landmarks (MediaPipe only) cho webcam
set -e

WEBCAM_INDEX="${WEBCAM_INDEX:-0}"
OUTPUT="${OUTPUT:-outputs}"

echo "📷  Chạy MediaPipe landmark detection từ webcam (index=${WEBCAM_INDEX})"
echo "   Nhấn 'q' trong cửa sổ OpenCV để dừng."
python src/preprocess/landmark_mediapipe.py \
    --input "${WEBCAM_INDEX}" \
    --output_dir "${OUTPUT}/landmarks2d"
