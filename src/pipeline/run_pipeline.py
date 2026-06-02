"""
Pipeline end-to-end: MediaPipe Landmarks → DECA 3D Reconstruction.

Stage 1 chạy trực tiếp trong môi trường hiện tại (deca_dhmt).
Stage 2 gọi DECA qua subprocess trong môi trường deca_env để tránh
xung đột phiên bản thư viện (torch 1.6 vs thư viện mới).

Sử dụng:
    conda activate deca_dhmt
    python src/pipeline/run_pipeline.py --input data/samples/test.jpg
    python src/pipeline/run_pipeline.py --input data/samples/test.mp4
    python src/pipeline/run_pipeline.py --input 0   # webcam
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Thêm project root vào PYTHONPATH
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Stage 1 import bình thường — chạy trong deca_dhmt
from src.utils.logger import get_logger, setup_root_logger
from src.utils.io import ensure_dir
from src.preprocess.landmark_mediapipe import process_image, process_video
# KHÔNG import bất cứ thứ gì từ src.recon.run_deca ở đây

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DECA_DHMT — Pipeline end-to-end: Landmarks + 3D Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn ảnh (.jpg/.png), video (.mp4), hoặc webcam index (vd: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Thư mục gốc lưu tất cả kết quả",
    )
    parser.add_argument(
        "--deca_root",
        type=Path,
        default=Path("external/DECA"),
        help="Đường dẫn tới DECA submodule",
    )
    parser.add_argument(
        "--deca_env",
        default="deca_env",
        help="Tên conda environment chứa DECA dependencies",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy DECA",
    )
    parser.add_argument(
        "--max_num_faces",
        type=int,
        default=1,
        help="Số khuôn mặt tối đa cho MediaPipe",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=None,
        help="Giới hạn cạnh lớn nhất trước khi chạy MediaPipe (vd: 1024).",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Chỉ xử lý mỗi N frame (video) để giảm tải.",
    )
    parser.add_argument(
        "--no_annotate",
        action="store_true",
        help="Không vẽ/ghi ảnh hoặc video annotate.",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Không ghi JSON landmarks (giảm I/O).",
    )
    parser.add_argument(
        "--deca_batch_size",
        type=int,
        default=1,
        help="Số ảnh chạy mỗi lần gọi DECA demo (batch).",
    )
    parser.add_argument(
        "--no_save_obj",
        action="store_true",
        help="Không lưu mesh .obj từ DECA.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Bật chế độ nhanh (giảm output và resize).",
    )
    parser.add_argument(
        "--skip_deca",
        action="store_true",
        help="Bỏ qua Stage 2 — chỉ chạy MediaPipe landmarks",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def _apply_fast_mode(
    max_image_size: Optional[int],
    frame_stride: int,
    save_annotated: bool,
    save_obj: bool,
    batch_size: int,
) -> Tuple[Optional[int], int, bool, bool, int]:
    if max_image_size is None:
        max_image_size = 1024
    if frame_stride == 1:
        frame_stride = 2
    if save_annotated:
        save_annotated = False
    if save_obj:
        save_obj = False
    if batch_size == 1:
        batch_size = 4
    return max_image_size, frame_stride, save_annotated, save_obj, batch_size


# ---------------------------------------------------------------------------
# Stage 1 — MediaPipe (chạy trong deca_dhmt, env hiện tại)
# ---------------------------------------------------------------------------
def _run_stage1(
    input_str: str,
    landmark_dir: Path,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    max_num_faces: int,
    max_image_size: Optional[int],
    save_annotated: bool,
    save_json_output: bool,
    frame_stride: int,
) -> bool:
    """Phát hiện landmarks bằng MediaPipe, lưu JSON vào landmark_dir."""
    logger.info("━" * 55)
    logger.info("STAGE 1 — MediaPipe Face Landmark Detection")
    logger.info("━" * 55)

    ensure_dir(landmark_dir)

    # Phân biệt webcam / video / ảnh
    try:
        webcam_index = int(input_str)
        logger.info(f"  Mode : Webcam (index={webcam_index})")
        return process_video(
            input_source=input_str,
            output_dir=landmark_dir,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            is_webcam=True,
            max_num_faces=max_num_faces,
            max_image_size=max_image_size,
            save_annotated=save_annotated,
            save_json_output=save_json_output,
            frame_stride=frame_stride,
        )
    except ValueError:
        pass

    input_path = Path(input_str)
    if not input_path.exists():
        logger.error(f"  File không tồn tại: {input_path}")
        return False

    ext = input_path.suffix.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if ext in image_exts:
        logger.info(f"  Mode : Ảnh tĩnh — {input_path.name}")
        return process_image(
            input_path=input_path,
            output_dir=landmark_dir,
            min_detection_confidence=min_detection_confidence,
            max_num_faces=max_num_faces,
            max_image_size=max_image_size,
            save_annotated=save_annotated,
            save_json_output=save_json_output,
        )
    elif ext in video_exts:
        logger.info(f"  Mode : Video — {input_path.name}")
        return process_video(
            input_source=str(input_path),
            output_dir=landmark_dir,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            is_webcam=False,
            max_num_faces=max_num_faces,
            max_image_size=max_image_size,
            save_annotated=save_annotated,
            save_json_output=save_json_output,
            frame_stride=frame_stride,
        )
    else:
        logger.error(f"  Định dạng không hỗ trợ: {ext}")
        return False


# ---------------------------------------------------------------------------
# Stage 2 — DECA (gọi qua subprocess vào conda env deca_env)
# ---------------------------------------------------------------------------
def _build_deca_cmd(
    deca_env: str,
    landmarks_dir: Path,
    output_dir: Path,
    deca_root: Path,
    device: str,
    batch_size: int,
    save_obj: bool,
) -> list[str]:
    """Tạo lệnh conda run để gọi run_deca.py trong deca_env."""
    deca_script = str(_ROOT / "src" / "recon" / "run_deca.py")
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", deca_env,
        "python", deca_script,
        "--landmarks_dir", str(landmarks_dir),
        "--output_dir",    str(output_dir),
        "--deca_root",     str(deca_root),
        "--device",        device,
    ]
    if batch_size > 1:
        cmd += ["--batch_size", str(batch_size)]
    if not save_obj:
        cmd += ["--no_save_obj"]
    return cmd


def _run_stage2(
    landmark_dir: Path,
    output_dir: Path,
    deca_root: Path,
    deca_env: str,
    device: str,
    input_str: str,
    batch_size: int,
    save_obj: bool,
) -> bool:
    """Gọi DECA qua subprocess trong deca_env."""
    logger.info("━" * 55)
    logger.info("STAGE 2 — DECA 3D Face Reconstruction (deca_env)")
    logger.info("━" * 55)

    # Bỏ qua DECA cho webcam — không có ảnh tĩnh để xử lý
    try:
        int(input_str)
        logger.warning("  Bỏ qua DECA cho webcam mode.")
        return True
    except ValueError:
        pass

    # Kiểm tra có JSON landmarks từ Stage 1 không
    json_files = list(landmark_dir.glob("*_landmarks.json"))
    if not json_files:
        logger.error(f"  Không có file JSON landmarks trong: {landmark_dir}")
        logger.error("  Stage 1 chưa chạy thành công hoặc sai --output.")
        return False

    logger.info(f"  Tìm thấy {len(json_files)} file landmarks")
    logger.info(f"  Gọi deca_env: conda run -n {deca_env} python src/recon/run_deca.py")

    cmd = _build_deca_cmd(
        deca_env      = deca_env,
        landmarks_dir = landmark_dir,
        output_dir    = output_dir,
        deca_root     = deca_root,
        device        = device,
        batch_size    = batch_size,
        save_obj      = save_obj,
    )

    logger.info(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, text=True)
    except FileNotFoundError:
        logger.error("  Không tìm thấy lệnh 'conda'.")
        logger.error("  Đảm bảo Conda đã được cài và có trong PATH.")
        return False
    except Exception as e:
        logger.error(f"  Lỗi không mong đợi khi gọi subprocess: {e}")
        return False

    if result.returncode != 0:
        logger.error(f"  DECA thất bại (exit code {result.returncode})")
        logger.error("  Kiểm tra:")
        logger.error("    1. conda env deca_env đã tạo chưa?  → bash setup.sh")
        logger.error("    2. DECA weights đã tải chưa?         → bash scripts/download_weights.sh")
        logger.error("    3. torch==1.6 + cudatoolkit=10.1 có khớp GPU không?")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    setup_root_logger()

    output_dir = args.output.resolve()
    deca_root  = args.deca_root.resolve()

    landmark_dir = output_dir / "landmarks2d"
    save_annotated = not args.no_annotate
    save_json_output = not args.no_json
    save_obj = not args.no_save_obj
    max_image_size = args.max_image_size
    frame_stride = args.frame_stride
    batch_size = max(1, args.deca_batch_size)

    if args.fast:
        max_image_size, frame_stride, save_annotated, save_obj, batch_size = _apply_fast_mode(
            max_image_size=max_image_size,
            frame_stride=frame_stride,
            save_annotated=save_annotated,
            save_obj=save_obj,
            batch_size=batch_size,
        )

    if not args.skip_deca and not save_json_output:
        logger.error(
            "Stage 2 yêu cầu JSON landmarks. Bỏ --no_json hoặc dùng --skip_deca để bỏ DECA."
        )
        sys.exit(1)

    # Tạo trước tất cả thư mục output
    for d in [landmark_dir,
              output_dir / "meshes",
              output_dir / "textures",
              output_dir / "renders",
              output_dir / "videos"]:
        ensure_dir(d)

    logger.info("=" * 55)
    logger.info("DECA_DHMT — Full Pipeline")
    logger.info("=" * 55)
    logger.info(f"Input      : {args.input}")
    logger.info(f"Output     : {output_dir}")
    logger.info(f"DECA root  : {deca_root}")
    logger.info(f"DECA env   : {args.deca_env}")
    logger.info(f"Device     : {args.device}")
    logger.info(f"Skip DECA  : {args.skip_deca}")
    logger.info(f"Max faces  : {args.max_num_faces}")
    logger.info(f"Max size   : {max_image_size}")
    logger.info(f"Stride     : {frame_stride}")
    logger.info(f"Annotate   : {save_annotated}")
    logger.info(f"Save JSON  : {save_json_output}")
    logger.info(f"DECA batch : {batch_size}")
    logger.info(f"Save OBJ   : {save_obj}")
    logger.info("=" * 55)

    pipeline_start = time.time()

    # ── Stage 1 ──────────────────────────────────────────
    t0 = time.time()
    stage1_ok = _run_stage1(
        input_str                = args.input,
        landmark_dir             = landmark_dir,
        min_detection_confidence = args.min_detection_confidence,
        min_tracking_confidence  = args.min_tracking_confidence,
        max_num_faces            = args.max_num_faces,
        max_image_size           = max_image_size,
        save_annotated           = save_annotated,
        save_json_output         = save_json_output,
        frame_stride             = frame_stride,
    )
    t1 = time.time() - t0

    if stage1_ok:
        logger.info(f"✅ Stage 1 hoàn thành ({t1:.2f}s)")
    else:
        logger.error(f"❌ Stage 1 thất bại ({t1:.2f}s) — dừng pipeline.")
        sys.exit(1)

    # ── Stage 2 ──────────────────────────────────────────
    if args.skip_deca:
        logger.info("⏭️  Stage 2 bị bỏ qua (--skip_deca).")
    else:
        t0 = time.time()
        stage2_ok = _run_stage2(
            landmark_dir = landmark_dir,
            output_dir   = output_dir,
            deca_root    = deca_root,
            deca_env     = args.deca_env,
            device       = args.device,
            input_str    = args.input,
            batch_size   = batch_size,
            save_obj     = save_obj,
        )
        t2 = time.time() - t0

        if stage2_ok:
            logger.info(f"✅ Stage 2 hoàn thành ({t2:.2f}s)")
        else:
            logger.error(f"❌ Stage 2 thất bại ({t2:.2f}s)")
            sys.exit(1)

    # ── Tổng kết ─────────────────────────────────────────
    total = time.time() - pipeline_start
    logger.info("=" * 55)
    logger.info(f"✅ Pipeline hoàn thành! Tổng thời gian: {total:.2f}s")
    logger.info(f"Kết quả tại: {output_dir}")


if __name__ == "__main__":
    main()