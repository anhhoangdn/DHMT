"""
3D Morphable Model (3DMM) fitting utilities.
TODO: Tích hợp với FLAME model hoặc BFM qua DECA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_3dmm_params(params_file: Path) -> Dict[str, Any]:
    """
    Tải tham số 3DMM từ file (output của DECA).

    Args:
        params_file: Đường dẫn file .npy hoặc .npz chứa tham số DECA.

    Returns:
        Dictionary chứa các tham số: shape, pose, expression, tex, cam, light.
    """
    # TODO: Implement loading of DECA output params
    if not params_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file tham số: {params_file}")
    logger.info(f"Tải tham số 3DMM từ: {params_file}")
    # DECA lưu dưới dạng .npy dict
    params = dict(np.load(str(params_file), allow_pickle=True).item())
    logger.info(f"Các tham số: {list(params.keys())}")
    return params


def decode_shape_params(shape_params: np.ndarray, shape_basis: np.ndarray) -> np.ndarray:
    """
    Tái tạo 3D mesh từ tham số hình dạng và PCA basis.

    Args:
        shape_params: Vector tham số hình dạng (shape_dim,).
        shape_basis: Ma trận cơ sở PCA (num_vertices*3, shape_dim).

    Returns:
        3D vertices (num_vertices, 3).
    """
    # TODO: Implement full shape decoding
    vertices_flat = shape_basis @ shape_params
    vertices = vertices_flat.reshape(-1, 3)
    return vertices
