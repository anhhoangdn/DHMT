"""
Căn chỉnh khuôn mặt (face alignment) dựa trên landmarks.
TODO: Tích hợp với landmarks từ MediaPipe để căn chỉnh trước khi đưa vào DECA.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tọa độ mắt chuẩn cho ảnh 224x224 (DECA standard)
LEFT_EYE_TARGET = (0.35, 0.40)
RIGHT_EYE_TARGET = (0.65, 0.40)
TARGET_SIZE = (224, 224)

# MediaPipe Face Mesh: indices cho mắt trái và phải
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def get_eye_center(
    landmarks_dict: dict,
    eye_indices: list,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    """
    Tính tâm mắt từ danh sách landmarks.

    Args:
        landmarks_dict: Dict chứa danh sách landmarks từ MediaPipe.
        eye_indices: Danh sách indices của các điểm quanh mắt.
        image_width: Chiều rộng ảnh.
        image_height: Chiều cao ảnh.

    Returns:
        Tọa độ tâm mắt (x_px, y_px).
    """
    points = landmarks_dict.get("landmarks", [])
    eye_points = [p for p in points if p["index"] in eye_indices]
    if not eye_points:
        raise ValueError(f"Không tìm thấy landmarks cho indices: {eye_indices}")
    x = np.mean([p["x_px"] for p in eye_points])
    y = np.mean([p["y_px"] for p in eye_points])
    return float(x), float(y)


def align_face(
    image: np.ndarray,
    left_eye: Tuple[float, float],
    right_eye: Tuple[float, float],
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """
    Căn chỉnh khuôn mặt dựa trên vị trí 2 mắt (affine transform).

    Args:
        image: Ảnh BGR gốc.
        left_eye: Tọa độ mắt trái (x, y) pixel.
        right_eye: Tọa độ mắt phải (x, y) pixel.
        target_size: Kích thước ảnh output.

    Returns:
        Ảnh đã căn chỉnh.
    """
    # TODO: Cài đặt affine alignment đầy đủ
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    aligned = cv2.resize(aligned, target_size)
    logger.debug(f"Căn chỉnh khuôn mặt: angle={angle:.2f}°, target_size={target_size}")
    return aligned
