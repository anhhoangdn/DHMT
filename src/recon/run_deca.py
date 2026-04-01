"""
Wrapper cho DECA inference.
Gọi DECA qua subprocess, KHÔNG tái cài đặt DECA internals.

Sử dụng:
    python src/recon/run_deca.py --input <input> --output <output_dir> --deca-root external/DECA

Yêu cầu:
    - DECA đã được cài đặt tại --deca-root (submodule init)
    - Pretrained weights đã tải về checkpoints/ của DECA
    - DECA dependencies đã được cài: pip install -r external/DECA/requirements.txt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Thêm thư mục gốc vào PYTHONPATH
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger
from src.utils.io import ensure_dir

logger = get_logger(__name__)

# Các script DECA có thể có (thứ tự ưu tiên)
DECA_DEMO_SCRIPTS = [
    "demos/demo_reconstruct.py",
    "demo.py",
    "demos/demo.py",
    "demo_reconstruct.py",
]

# Tên file/thư mục để xác nhận DECA hợp lệ
DECA_REQUIRED_FILES = [
    "decalib",
    "README.md",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper cho DECA 3D face reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn ảnh hoặc thư mục ảnh đầu vào",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Thư mục lưu kết quả DECA",
    )
    parser.add_argument(
        "--deca-root",
        type=Path,
        default=Path("external/DECA"),
        dest="deca_root",
        help="Đường dẫn tới thư mục DECA (submodule)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy inference (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--save-obj",
        action="store_true",
        default=True,
        help="Lưu file .obj mesh",
    )
    return parser.parse_args()


def validate_deca_installation(deca_root: Path) -> Optional[Path]:
    """
    Kiểm tra DECA có tồn tại và hợp lệ không.

    Args:
        deca_root: Thư mục DECA.

    Returns:
        Đường dẫn script demo nếu tìm thấy, None nếu không.
    """
    if not deca_root.exists():
        logger.error(f"❌ Không tìm thấy thư mục DECA: {deca_root}")
        logger.error("")
        logger.error("Để thiết lập DECA, chạy các lệnh sau:")
        logger.error("  git submodule add https://github.com/yfeng95/DECA.git external/DECA")
        logger.error("  git submodule update --init --recursive")
        logger.error("  pip install -r external/DECA/requirements.txt")
        logger.error("  bash scripts/download_weights.sh")
        return None

    # Kiểm tra các file cần thiết
    for required in DECA_REQUIRED_FILES:
        if not (deca_root / required).exists():
            logger.warning(f"⚠️  Thiếu file/thư mục trong DECA: {required}")
            logger.warning("DECA submodule có thể chưa được khởi tạo đúng cách.")
            logger.warning("Chạy: git submodule update --init --recursive")

    # Tìm script demo
    demo_script: Optional[Path] = None
    for script_rel in DECA_DEMO_SCRIPTS:
        candidate = deca_root / script_rel
        if candidate.exists():
            demo_script = candidate
            logger.info(f"✅ Tìm thấy DECA demo script: {demo_script}")
            break

    if demo_script is None:
        logger.error("❌ Không tìm thấy script demo của DECA.")
        logger.error(f"Đã tìm kiếm trong: {[str(deca_root / s) for s in DECA_DEMO_SCRIPTS]}")
        logger.error("")
        logger.error("Kiểm tra lại submodule:")
        logger.error("  git submodule update --init --recursive")
        logger.error("  ls external/DECA/demos/")
        return None

    return demo_script


def check_deca_weights(deca_root: Path) -> bool:
    """
    Kiểm tra pretrained weights của DECA.

    Args:
        deca_root: Thư mục DECA.

    Returns:
        True nếu weights tồn tại.
    """
    weight_candidates = [
        deca_root / "data" / "deca_model.tar",
        deca_root / "data" / "generic_model.pkl",
        deca_root / "checkpoints" / "deca_model.tar",
    ]
    for w in weight_candidates:
        if w.exists():
            logger.info(f"✅ Tìm thấy DECA weights: {w}")
            return True

    logger.warning("⚠️  Không tìm thấy DECA pretrained weights!")
    logger.warning("Tải weights theo hướng dẫn trong scripts/download_weights.sh")
    logger.warning("hoặc xem: https://github.com/yfeng95/DECA#getting-started")
    return False


def run_deca(
    input_path: Path,
    output_dir: Path,
    deca_root: Path,
    demo_script: Path,
    device: str = "cuda",
    save_obj: bool = True,
) -> int:
    """
    Chạy DECA inference thông qua subprocess.

    Args:
        input_path: Ảnh hoặc thư mục ảnh đầu vào.
        output_dir: Thư mục lưu kết quả.
        deca_root: Thư mục gốc của DECA.
        demo_script: Đường dẫn tới script demo.
        device: 'cuda' hoặc 'cpu'.
        save_obj: Lưu file .obj không.

    Returns:
        Return code của process (0 = thành công).
    """
    ensure_dir(output_dir)

    cmd: List[str] = [
        sys.executable,
        str(demo_script),
        "--inputpath", str(input_path),
        "--savefolder", str(output_dir),
        "--device", device,
    ]
    if save_obj:
        cmd.extend(["--saveObj", "True"])

    logger.info(f"Chạy DECA: {' '.join(cmd)}")
    logger.info(f"Thư mục làm việc: {deca_root}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(deca_root),
            capture_output=False,
            text=True,
        )
    except FileNotFoundError as e:
        logger.error(f"❌ Lỗi khi chạy DECA: {e}")
        logger.error("Đảm bảo Python interpreter và DECA dependencies đã được cài đặt.")
        return 1
    except Exception as e:
        logger.error(f"❌ Lỗi không mong đợi: {e}")
        return 1

    if result.returncode != 0:
        logger.error(f"❌ DECA thất bại với return code: {result.returncode}")
        logger.error("Kiểm tra:")
        logger.error("  1. DECA weights đã được tải về chưa?")
        logger.error("  2. DECA dependencies đã cài chưa? (pip install -r external/DECA/requirements.txt)")
        logger.error("  3. CUDA/PyTorch có tương thích không?")
    else:
        logger.info(f"✅ DECA hoàn thành! Kết quả lưu tại: {output_dir}")

    return result.returncode


def main() -> None:
    args = _parse_args()

    deca_root = args.deca_root.resolve()
    input_path = Path(args.input).resolve()
    output_dir = args.output.resolve()

    logger.info("=" * 60)
    logger.info("DECA 3D Face Reconstruction - DECA_DHMT")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"DECA root: {deca_root}")
    logger.info(f"Device: {args.device}")

    if not input_path.exists():
        logger.error(f"❌ File/thư mục đầu vào không tồn tại: {input_path}")
        sys.exit(1)

    demo_script = validate_deca_installation(deca_root)
    if demo_script is None:
        sys.exit(1)

    check_deca_weights(deca_root)

    deca_output_dir = output_dir / "deca"
    ensure_dir(deca_output_dir)

    return_code = run_deca(
        input_path=input_path,
        output_dir=deca_output_dir,
        deca_root=deca_root,
        demo_script=demo_script,
        device=args.device,
        save_obj=args.save_obj,
    )

    sys.exit(return_code)


if __name__ == "__main__":
    main()
