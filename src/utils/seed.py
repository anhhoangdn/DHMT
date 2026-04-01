"""
Thiết lập random seed để tái tạo kết quả.
"""
from __future__ import annotations

import random
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Đặt random seed cho Python, NumPy, và PyTorch (nếu có).

    Args:
        seed: Giá trị seed (mặc định 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Đã đặt seed (PyTorch + NumPy + Python): {seed}")
    except ImportError:
        logger.info(f"Đã đặt seed (NumPy + Python, PyTorch không có): {seed}")
