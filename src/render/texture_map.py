"""
Texture mapping cho 3D face mesh.
TODO: Implement UV unwrapping và texture projection từ ảnh gốc.
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_texture(texture_path: Path) -> np.ndarray:
    """
    Tải texture image.

    Args:
        texture_path: Đường dẫn file ảnh texture.

    Returns:
        Ảnh texture dạng numpy array (H, W, 3) BGR.
    """
    img = cv2.imread(str(texture_path))
    if img is None:
        raise FileNotFoundError(f"Không thể tải texture: {texture_path}")
    logger.info(f"Tải texture: {texture_path} {img.shape}")
    return img


def save_texture(texture: np.ndarray, output_path: Path) -> None:
    """
    Lưu texture ra file.

    Args:
        texture: Ảnh texture (H, W, 3) BGR.
        output_path: Đường dẫn lưu.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), texture)
    logger.info(f"Đã lưu texture: {output_path}")


def apply_uv_mapping(
    image: np.ndarray,
    uvs: np.ndarray,
    texture_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Ánh xạ màu từ ảnh nguồn lên texture map theo UV coordinates.

    Args:
        image: Ảnh nguồn BGR.
        uvs: UV coordinates (N, 2), giá trị trong [0, 1].
        texture_size: Kích thước texture output (H, W).

    Returns:
        Texture map (H, W, 3) BGR.
    """
    # TODO: Implement proper UV mapping với rasterization
    h, w = texture_size
    texture = np.zeros((h, w, 3), dtype=np.uint8)
    img_h, img_w = image.shape[:2]

    for uv in uvs:
        u, v = float(uv[0]), float(uv[1])
        # Clamp về [0, 1]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        # Lấy màu từ ảnh gốc
        src_x = int(u * (img_w - 1))
        src_y = int((1 - v) * (img_h - 1))  # Flip Y
        tex_x = int(u * (w - 1))
        tex_y = int((1 - v) * (h - 1))
        texture[tex_y, tex_x] = image[src_y, src_x]

    logger.debug(f"UV mapping hoàn thành: texture size {texture_size}")
    return texture
