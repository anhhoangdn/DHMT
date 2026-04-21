"""
DECA inference wrapper — chạy độc lập trong môi trường deca_env.

Được gọi bởi src/pipeline/run_pipeline.py qua subprocess:
    conda run -n deca_env python src/recon/run_deca.py \
        --landmarks_dir outputs/landmarks2d \
        --output_dir    outputs \
        --deca_root     external/DECA \
        --device        cuda

KHÔNG import bất kỳ thứ gì từ src/ (tránh cross-env dependency).
Giao tiếp với Stage 1 qua file JSON trong --landmarks_dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logger tự định nghĩa — KHÔNG import từ src.utils để tránh cross-env
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECA_DEMO_SCRIPTS = [
    "demos/demo_reconstruct.py",
    "demo.py",
    "demos/demo.py",
    "demo_reconstruct.py",
]

DECA_WEIGHT_CANDIDATES = [
    "data/deca_model.tar",
    "data/generic_model.pkl",
    "checkpoints/deca_model.tar",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DECA 3D face reconstruction — chạy trong deca_env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--landmarks_dir",
        type=Path,
        required=True,
        help="Thư mục chứa JSON landmarks từ Stage 1 (outputs/landmarks2d/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Thư mục gốc lưu kết quả (meshes/, textures/, renders/)",
    )
    parser.add_argument(
        "--deca_root",
        type=Path,
        default=Path("external/DECA"),
        help="Đường dẫn tới DECA submodule",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy inference",
    )
    parser.add_argument(
        "--save_obj",
        action="store_true",
        default=True,
        help="Lưu file .obj mesh",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_deca(deca_root: Path) -> Optional[Path]:
    """Kiểm tra DECA tồn tại và trả về đường dẫn demo script."""
    if not deca_root.exists():
        logger.error(f"Không tìm thấy DECA: {deca_root}")
        logger.error("Chạy: git submodule update --init --recursive")
        return None

    if not (deca_root / "decalib").exists():
        logger.error("Thiếu thư mục decalib/ — submodule chưa init đúng.")
        logger.error("Chạy: git submodule update --init --recursive")
        return None

    for script_rel in DECA_DEMO_SCRIPTS:
        candidate = deca_root / script_rel
        if candidate.exists():
            logger.info(f"Tìm thấy DECA script: {candidate}")
            return candidate

    logger.error("Không tìm thấy demo script trong DECA.")
    logger.error(f"Đã tìm: {[str(deca_root / s) for s in DECA_DEMO_SCRIPTS]}")
    return None


def _check_weights(deca_root: Path) -> bool:
    """Kiểm tra pretrained weights có tồn tại không."""
    for rel in DECA_WEIGHT_CANDIDATES:
        if (deca_root / rel).exists():
            logger.info(f"Tìm thấy weights: {deca_root / rel}")
            return True
    logger.warning("Không tìm thấy DECA weights!")
    logger.warning("Chạy: bash scripts/download_weights.sh")
    return False


# ---------------------------------------------------------------------------
# Đọc JSON landmarks từ Stage 1
# ---------------------------------------------------------------------------
def _load_image_paths_from_landmarks(landmarks_dir: Path) -> list[str]:
    """
    Đọc JSON landmarks từ Stage 1.
    Format: {"source": "...", "faces": [...]}
    """
    json_files = sorted(landmarks_dir.glob("*_landmarks.json"))

    if not json_files:
        logger.error(f"Không có file *_landmarks.json trong: {landmarks_dir}")
        logger.error("Stage 1 (MediaPipe) chưa chạy hoặc sai --landmarks_dir.")
        return []

    image_paths = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)

            # ── Đọc đúng key "source" từ landmark_mediapipe.py ──
            img_path = data.get("source", "")
            if not img_path:
                logger.warning(f"JSON thiếu key 'source': {jf.name}")
                continue

            # Bỏ qua JSON của video (có key "frames") — chỉ lấy ảnh tĩnh
            if "frames" in data:
                logger.info(f"  Bỏ qua (video JSON): {jf.name}")
                continue

            # Kiểm tra có detect được mặt không
            faces = data.get("faces", [])
            if not faces:
                logger.warning(f"  Không có khuôn mặt trong: {jf.name}")
                continue

            if not Path(img_path).exists():
                logger.warning(f"  Ảnh gốc không tồn tại: {img_path}")
                continue

            image_paths.append(img_path)
            logger.info(f"  Đọc OK: {jf.name} → {img_path} ({len(faces)} mặt)")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Lỗi đọc {jf.name}: {e}")

    logger.info(f"Tổng ảnh cần xử lý: {len(image_paths)}")
    return image_paths


# ---------------------------------------------------------------------------
# Chạy DECA cho từng ảnh
# ---------------------------------------------------------------------------
def _run_deca_single(
    image_path: str,
    output_dir: Path,
    deca_root: Path,
    demo_script: Path,
    device: str,
    save_obj: bool,
) -> int:
    """Gọi DECA demo script cho 1 ảnh qua subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── THÊM DÒNG NÀY: đổi sang absolute path ──
    image_path_abs = str(Path(image_path).resolve())

    cmd = [
        sys.executable,
        str(demo_script),
        "--inputpath", image_path_abs,   # ← dùng absolute path
        "--savefolder", str(output_dir),
        "--device", device,
    ]
    if save_obj:
        cmd += ["--saveObj", "True"]

    logger.info(f"Chạy: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(deca_root),
            text=True,
        )
        return result.returncode
    except FileNotFoundError as e:
        logger.error(f"Lỗi FileNotFoundError: {e}")
        return 1
    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {e}")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    landmarks_dir = args.landmarks_dir.resolve()
    output_dir    = args.output_dir.resolve()
    deca_root     = args.deca_root.resolve()

    logger.info("=" * 55)
    logger.info("Stage 2 — DECA 3D Reconstruction (deca_env)")
    logger.info("=" * 55)
    logger.info(f"landmarks_dir : {landmarks_dir}")
    logger.info(f"output_dir    : {output_dir}")
    logger.info(f"deca_root     : {deca_root}")
    logger.info(f"device        : {args.device}")

    # 1. Validate DECA
    demo_script = _validate_deca(deca_root)
    if demo_script is None:
        sys.exit(1)

    # 2. Kiểm tra weights (warning only, không exit)
    _check_weights(deca_root)

    # 3. Đọc danh sách ảnh từ JSON landmarks Stage 1
    image_paths = _load_image_paths_from_landmarks(landmarks_dir)
    if not image_paths:
        sys.exit(1)

    # 4. Chạy DECA cho từng ảnh
    deca_out = output_dir / "deca"   # DECA tự tạo cấu trúc con bên trong
    failed   = []

    for img_path in image_paths:
        basename = Path(img_path).stem
        logger.info(f"\nXử lý: {basename}")

        rc = _run_deca_single(
            image_path  = img_path,
            output_dir  = deca_out / basename,
            deca_root   = deca_root,
            demo_script = demo_script,
            device      = args.device,
            save_obj    = args.save_obj,
        )

        if rc != 0:
            logger.error(f"DECA thất bại: {basename} (code {rc})")
            failed.append(basename)
        else:
            logger.info(f"✓ {basename} → {deca_out / basename}")

    # 5. Tổng kết
    logger.info("\n" + "=" * 55)
    total   = len(image_paths)
    success = total - len(failed)
    logger.info(f"Kết quả: {success}/{total} ảnh thành công")

    if failed:
        logger.warning(f"Thất bại: {failed}")
        logger.warning("Kiểm tra:")
        logger.warning("  1. DECA weights đã tải chưa?  → bash scripts/download_weights.sh")
        logger.warning("  2. DECA deps đã cài chưa?     → pip install -r external/DECA/requirements.txt")
        logger.warning("  3. CUDA/PyTorch tương thích?  → torch==1.8.0 + cudatoolkit=10.2")
        sys.exit(1)

    logger.info(f"Kết quả lưu tại: {deca_out}")
    sys.exit(0)


if __name__ == "__main__":
    main()