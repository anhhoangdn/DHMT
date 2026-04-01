"""
Hậu xử lý kết quả tái tạo 3D từ DECA.
TODO: Implement mesh smoothing, texture refinement.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def smooth_mesh(vertices: np.ndarray, faces: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Làm mịn mesh bằng Laplacian smoothing.

    Args:
        vertices: Mảng vertices (N, 3).
        faces: Mảng faces (F, 3) - indices vào vertices.
        iterations: Số lần lặp smoothing.

    Returns:
        Vertices đã làm mịn.
    """
    # TODO: Implement Laplacian smoothing
    logger.info(f"Laplacian smoothing: {iterations} lần lặp, {len(vertices)} vertices")
    smoothed = vertices.copy()
    for _ in range(iterations):
        # Placeholder: chưa implement đầy đủ
        pass
    return smoothed


def normalize_mesh(vertices: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa mesh về center và unit scale.

    Args:
        vertices: Mảng vertices (N, 3).

    Returns:
        Vertices đã chuẩn hóa.
    """
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    scale = np.max(np.abs(vertices_centered))
    if scale > 0:
        vertices_normalized = vertices_centered / scale
    else:
        vertices_normalized = vertices_centered
    logger.debug(f"Chuẩn hóa mesh: center={center}, scale={scale:.4f}")
    return vertices_normalized
