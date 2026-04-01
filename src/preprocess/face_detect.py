"""
Phát hiện khuôn mặt trong ảnh/video.
TODO: Tích hợp thêm các detector khác (RetinaFace, MTCNN).
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_faces_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Phát hiện khuôn mặt bằng OpenCV Haar Cascade.

    Args:
        image: Ảnh BGR dạng numpy array.

    Returns:
        Danh sách bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        logger.warning("Không phát hiện được khuôn mặt nào.")
        return []
    logger.info(f"Phát hiện {len(faces)} khuôn mặt.")
    return [tuple(f) for f in faces]


def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.2,
) -> np.ndarray:
    """
    Cắt vùng khuôn mặt từ ảnh với padding.

    Args:
        image: Ảnh gốc.
        bbox: Bounding box (x, y, w, h).
        padding: Tỉ lệ padding thêm vào mỗi cạnh.

    Returns:
        Ảnh khuôn mặt đã cắt.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return image[y1:y2, x1:x2]
