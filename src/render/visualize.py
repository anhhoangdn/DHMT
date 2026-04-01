"""
Visualization cho landmarks và 3D mesh.
TODO: Tích hợp với Open3D hoặc Pyrender để render 3D.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def visualize_landmarks_2d(
    image: np.ndarray,
    landmarks: List[dict],
    output_path: Optional[Path] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """
    Vẽ 2D landmarks lên ảnh.

    Args:
        image: Ảnh BGR.
        landmarks: Danh sách dict landmarks với keys 'x_px', 'y_px'.
        output_path: Đường dẫn lưu ảnh kết quả (tùy chọn).
        color: Màu BGR cho các điểm.
        radius: Bán kính điểm.

    Returns:
        Ảnh đã vẽ landmarks.
    """
    vis_image = image.copy()
    for lm in landmarks:
        x, y = int(lm["x_px"]), int(lm["y_px"])
        cv2.circle(vis_image, (x, y), radius, color, -1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Đã lưu ảnh landmarks: {output_path}")

    return vis_image


def plot_landmarks_3d(
    landmarks: List[dict],
    output_path: Optional[Path] = None,
) -> None:
    """
    Vẽ landmarks 3D bằng matplotlib.

    Args:
        landmarks: Danh sách dict với keys 'x', 'y', 'z'.
        output_path: Đường dẫn lưu hình (tùy chọn).
    """
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    zs = [lm["z"] for lm in landmarks]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=1)
    ax.set_title("Face Landmarks 3D - DECA_DHMT")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu hình landmarks 3D: {output_path}")
        plt.close()
    else:
        plt.show()
