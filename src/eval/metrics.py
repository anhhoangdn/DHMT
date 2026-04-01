"""
Các metrics đánh giá chất lượng tái tạo 3D face.
TODO: Implement NME (Normalized Mean Error), 3D landmark error.
"""
from __future__ import annotations

import numpy as np
from typing import Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_nme(
    pred_landmarks: np.ndarray,
    gt_landmarks: np.ndarray,
    normalize_factor: float,
) -> float:
    """
    Tính Normalized Mean Error (NME) cho face landmarks.

    Args:
        pred_landmarks: Landmarks dự đoán (N, 2) hoặc (N, 3).
        gt_landmarks: Landmarks ground truth (N, 2) hoặc (N, 3).
        normalize_factor: Hệ số chuẩn hóa (vd: inter-ocular distance).

    Returns:
        NME value.
    """
    if pred_landmarks.shape != gt_landmarks.shape:
        raise ValueError(
            f"Kích thước không khớp: pred={pred_landmarks.shape}, gt={gt_landmarks.shape}"
        )
    error = np.linalg.norm(pred_landmarks - gt_landmarks, axis=-1).mean()
    nme = error / normalize_factor
    logger.debug(f"NME = {nme:.4f}")
    return float(nme)


def compute_3d_reconstruction_error(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
) -> Dict[str, float]:
    """
    Tính các metrics lỗi tái tạo 3D.

    Args:
        pred_vertices: Vertices dự đoán (N, 3).
        gt_vertices: Vertices ground truth (N, 3).

    Returns:
        Dict chứa MSE, MAE, max_error, RMSE.
    """
    diff = pred_vertices - gt_vertices
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))
    max_error = float(np.max(abs_diff))
    metrics = {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "rmse": float(np.sqrt(mse)),
    }
    logger.info(f"3D Error: MSE={mse:.6f}, MAE={mae:.6f}, MaxErr={max_error:.6f}")
    return metrics


def compute_psnr(pred_img: np.ndarray, gt_img: np.ndarray) -> float:
    """
    Tính Peak Signal-to-Noise Ratio (PSNR) cho ảnh texture.

    Args:
        pred_img: Ảnh dự đoán (H, W, 3), giá trị [0, 255].
        gt_img: Ảnh ground truth (H, W, 3).

    Returns:
        PSNR value (dB).
    """
    mse = np.mean((pred_img.astype(float) - gt_img.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    logger.debug(f"PSNR = {psnr:.2f} dB")
    return float(psnr)
