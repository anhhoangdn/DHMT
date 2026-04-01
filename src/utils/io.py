"""
Các tiện ích I/O cho DECA_DHMT.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Tạo thư mục nếu chưa tồn tại.

    Args:
        path: Đường dẫn thư mục cần tạo.

    Returns:
        Path object của thư mục.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Lưu dữ liệu ra file JSON.

    Args:
        data: Dữ liệu cần lưu (phải JSON-serializable).
        path: Đường dẫn file JSON.
        indent: Số khoảng trắng indent (mặc định 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Đã lưu JSON: {path}")


def load_json(path: Union[str, Path]) -> Any:
    """
    Tải dữ liệu từ file JSON.

    Args:
        path: Đường dẫn file JSON.

    Returns:
        Dữ liệu đã tải.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Đã tải JSON: {path}")
    return data


def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
    """
    Lưu numpy array ra file .npy.

    Args:
        array: Numpy array cần lưu.
        path: Đường dẫn file .npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    logger.debug(f"Đã lưu numpy array: {path} shape={array.shape}")


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """
    Tải numpy array từ file .npy.

    Args:
        path: Đường dẫn file .npy.

    Returns:
        Numpy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file numpy: {path}")
    arr = np.load(str(path), allow_pickle=True)
    logger.debug(f"Đã tải numpy: {path} shape={arr.shape}")
    return arr


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Sao chép file.

    Args:
        src: Đường dẫn nguồn.
        dst: Đường dẫn đích.

    Returns:
        Đường dẫn file đích.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    logger.debug(f"Đã sao chép: {src} -> {dst}")
    return dst


def list_image_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    """
    Liệt kê tất cả file ảnh trong thư mục.

    Args:
        directory: Thư mục cần liệt kê.
        extensions: Danh sách extension (mặc định: jpg, jpeg, png, bmp).

    Returns:
        Danh sách Path của các file ảnh.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Thư mục không tồn tại: {directory}")
        return []
    files = sorted([
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])
    logger.info(f"Tìm thấy {len(files)} ảnh trong {directory}")
    return files


def get_output_stem(input_path: Union[str, Path]) -> str:
    """
    Lấy stem (tên file không có extension) từ đường dẫn input.

    Args:
        input_path: Đường dẫn file input.

    Returns:
        Stem của file.
    """
    return Path(input_path).stem
