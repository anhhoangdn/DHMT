"""
Benchmark toàn bộ pipeline tái tạo 3D.
TODO: Chạy trên dataset chuẩn (300W, AFLW, etc.).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Any, Callable

import numpy as np
from src.utils.logger import get_logger
from src.utils.io import save_json

logger = get_logger(__name__)


def benchmark_pipeline(
    test_images: List[Path],
    output_dir: Path,
    run_fn: Callable[[Path], None],
) -> Dict[str, Any]:
    """
    Chạy benchmark trên danh sách ảnh test.

    Args:
        test_images: Danh sách đường dẫn ảnh.
        output_dir: Thư mục lưu kết quả benchmark.
        run_fn: Callable nhận path ảnh và chạy pipeline.

    Returns:
        Dict chứa kết quả benchmark.
    """
    results = []
    total_time = 0.0

    for img_path in test_images:
        logger.info(f"Benchmark: {img_path.name}")
        start = time.time()
        try:
            run_fn(img_path)
            elapsed = time.time() - start
            results.append({"file": str(img_path), "time_s": elapsed, "status": "success"})
            total_time += elapsed
        except Exception as e:
            results.append({"file": str(img_path), "time_s": 0.0, "status": f"error: {e}"})

    summary = {
        "total_images": len(test_images),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "avg_time_s": total_time / max(1, len(test_images)),
        "total_time_s": total_time,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "benchmark_results.json")
    logger.info(
        f"Benchmark hoàn thành: {summary['successful']}/{summary['total_images']} thành công,"
        f" avg={summary['avg_time_s']:.2f}s"
    )
    return summary
