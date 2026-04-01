"""
Pipeline end-to-end: MediaPipe Landmarks -> DECA 3D Reconstruction.

Sử dụng:
    python src/pipeline/run_pipeline.py --input <path> --output <output_dir> --deca-root external/DECA

Ví dụ:
    # Ảnh tĩnh
    python src/pipeline/run_pipeline.py --input data/samples/test.jpg
    # Video
    python src/pipeline/run_pipeline.py --input data/samples/test.mp4
    # Webcam
    python src/pipeline/run_pipeline.py --input 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger, setup_root_logger
from src.utils.io import ensure_dir
from src.preprocess.landmark_mediapipe import process_image, process_video
from src.recon.run_deca import validate_deca_installation, check_deca_weights, run_deca

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DECA_DHMT - Pipeline end-to-end: Landmarks + 3D Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn ảnh, video, hoặc chỉ số webcam (vd: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Thư mục gốc lưu tất cả kết quả",
    )
    parser.add_argument(
        "--deca-root",
        type=Path,
        default=Path("external/DECA"),
        dest="deca_root",
        help="Đường dẫn tới DECA submodule",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy DECA (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--skip-deca",
        action="store_true",
        help="Bỏ qua bước DECA (chỉ chạy MediaPipe landmarks)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        dest="min_detection_confidence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        dest="min_tracking_confidence",
    )
    return parser.parse_args()


def run_landmark_stage(
    input_str: str,
    output_dir: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> bool:
    """
    Stage 1: Phát hiện landmarks bằng MediaPipe.

    Args:
        input_str: Đường dẫn file hoặc webcam index.
        output_dir: Thư mục lưu landmarks.
        min_detection_confidence: Ngưỡng detection.
        min_tracking_confidence: Ngưỡng tracking.

    Returns:
        True nếu thành công.
    """
    logger.info("━" * 60)
    logger.info("STAGE 1: MediaPipe Face Landmark Detection")
    logger.info("━" * 60)

    ensure_dir(output_dir)

    try:
        webcam_index = int(input_str)
        logger.info(f"  Mode: Webcam (index={webcam_index})")
        return process_video(
            input_source=input_str,
            output_dir=output_dir,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            is_webcam=True,
        )
    except ValueError:
        input_path = Path(input_str)
        if not input_path.exists():
            logger.error(f"  ❌ File không tồn tại: {input_path}")
            return False

        ext = input_path.suffix.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if ext in image_exts:
            logger.info(f"  Mode: Ảnh tĩnh - {input_path.name}")
            return process_image(
                input_path=input_path,
                output_dir=output_dir,
                min_detection_confidence=min_detection_confidence,
            )
        elif ext in video_exts:
            logger.info(f"  Mode: Video - {input_path.name}")
            return process_video(
                input_source=str(input_path),
                output_dir=output_dir,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                is_webcam=False,
            )
        else:
            logger.error(f"  ❌ Định dạng không hỗ trợ: {ext}")
            return False


def run_deca_stage(
    input_str: str,
    output_dir: Path,
    deca_root: Path,
    device: str = "cuda",
) -> bool:
    """
    Stage 2: DECA 3D Face Reconstruction.

    Args:
        input_str: Đường dẫn file đầu vào.
        output_dir: Thư mục lưu kết quả DECA.
        deca_root: Thư mục DECA.
        device: 'cuda' hoặc 'cpu'.

    Returns:
        True nếu thành công.
    """
    logger.info("━" * 60)
    logger.info("STAGE 2: DECA 3D Face Reconstruction")
    logger.info("━" * 60)

    # Không chạy DECA trên webcam
    try:
        int(input_str)
        logger.warning("  ⚠️  Bỏ qua DECA cho webcam mode (chỉ xử lý ảnh/video).")
        return True
    except ValueError:
        pass

    input_path = Path(input_str)

    demo_script = validate_deca_installation(deca_root)
    if demo_script is None:
        return False

    check_deca_weights(deca_root)

    deca_output_dir = output_dir / "deca"
    ensure_dir(deca_output_dir)

    return_code = run_deca(
        input_path=input_path,
        output_dir=deca_output_dir,
        deca_root=deca_root,
        demo_script=demo_script,
        device=device,
    )
    return return_code == 0


def main() -> None:
    args = _parse_args()
    setup_root_logger()

    output_dir = args.output.resolve()
    deca_root = args.deca_root.resolve()

    # Tạo thư mục output
    landmark_dir = output_dir / "landmarks2d"
    ensure_dir(landmark_dir)
    ensure_dir(output_dir / "meshes")
    ensure_dir(output_dir / "textures")
    ensure_dir(output_dir / "renders")
    ensure_dir(output_dir / "videos")

    logger.info("=" * 60)
    logger.info("🚀 DECA_DHMT - Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input      : {args.input}")
    logger.info(f"Output     : {output_dir}")
    logger.info(f"DECA root  : {deca_root}")
    logger.info(f"Device     : {args.device}")
    logger.info(f"Skip DECA  : {args.skip_deca}")
    logger.info("=" * 60)

    pipeline_start = time.time()
    success = True

    # Stage 1: Landmarks
    stage1_start = time.time()
    stage1_ok = run_landmark_stage(
        input_str=args.input,
        output_dir=landmark_dir,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    stage1_time = time.time() - stage1_start

    if stage1_ok:
        logger.info(f"  ✅ Stage 1 hoàn thành ({stage1_time:.2f}s)")
    else:
        logger.error(f"  ❌ Stage 1 thất bại ({stage1_time:.2f}s)")
        success = False

    # Stage 2: DECA (bỏ qua nếu --skip-deca)
    if not args.skip_deca:
        stage2_start = time.time()
        stage2_ok = run_deca_stage(
            input_str=args.input,
            output_dir=output_dir,
            deca_root=deca_root,
            device=args.device,
        )
        stage2_time = time.time() - stage2_start

        if stage2_ok:
            logger.info(f"  ✅ Stage 2 hoàn thành ({stage2_time:.2f}s)")
        else:
            logger.error(f"  ❌ Stage 2 thất bại ({stage2_time:.2f}s)")
            success = False
    else:
        logger.info("  ⏭️  Stage 2 (DECA) bị bỏ qua.")

    total_time = time.time() - pipeline_start
    logger.info("=" * 60)
    if success:
        logger.info(f"✅ Pipeline hoàn thành! ({total_time:.2f}s)")
        logger.info(f"Kết quả tại: {output_dir}")
    else:
        logger.error(f"❌ Pipeline thất bại sau {total_time:.2f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
