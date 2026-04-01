"""
Phát hiện face landmarks bằng MediaPipe Face Mesh.

Sử dụng:
    python src/preprocess/landmark_mediapipe.py --input <path_or_webcam_index> --output_dir outputs/landmarks2d

Ví dụ:
    # Ảnh tĩnh
    python src/preprocess/landmark_mediapipe.py --input data/samples/test.jpg --output_dir outputs/landmarks2d
    # Video
    python src/preprocess/landmark_mediapipe.py --input data/samples/test.mp4 --output_dir outputs/landmarks2d
    # Webcam
    python src/preprocess/landmark_mediapipe.py --input 0 --output_dir outputs/landmarks2d
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH khi chạy trực tiếp
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger
from src.utils.io import ensure_dir, save_json

logger = get_logger(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phát hiện face landmarks bằng MediaPipe Face Mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn ảnh, video, hoặc chỉ số webcam (vd: 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/landmarks2d"),
        help="Thư mục lưu kết quả landmarks",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Ngưỡng tin cậy tối thiểu cho face detection",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Ngưỡng tin cậy tối thiểu cho face tracking",
    )
    parser.add_argument(
        "--max_num_faces",
        type=int,
        default=1,
        help="Số khuôn mặt tối đa cần detect",
    )
    return parser.parse_args()


def landmarks_to_dict(face_landmarks, image_width: int, image_height: int) -> Dict[str, Any]:
    """Chuyển đổi MediaPipe landmarks sang dict JSON-serializable."""
    points = []
    for idx, lm in enumerate(face_landmarks.landmark):
        points.append({
            "index": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "x_px": int(lm.x * image_width),
            "y_px": int(lm.y * image_height),
        })
    return {"landmarks": points, "num_landmarks": len(points)}


def draw_landmarks_on_image(image: np.ndarray, face_landmarks) -> np.ndarray:
    """Vẽ landmarks lên ảnh và trả về ảnh đã annotate."""
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    return annotated


def process_image(
    input_path: Path,
    output_dir: Path,
    min_detection_confidence: float = 0.5,
    max_num_faces: int = 1,
) -> bool:
    """
    Xử lý ảnh tĩnh: phát hiện landmarks và lưu JSON + ảnh annotate.

    Args:
        input_path: Đường dẫn ảnh đầu vào.
        output_dir: Thư mục lưu kết quả.
        min_detection_confidence: Ngưỡng tin cậy.
        max_num_faces: Số mặt tối đa.

    Returns:
        True nếu phát hiện được ít nhất 1 khuôn mặt.
    """
    ensure_dir(output_dir)
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        logger.error(f"Không thể đọc ảnh: {input_path}")
        return False

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    stem = input_path.stem

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        logger.warning(f"Không phát hiện được khuôn mặt trong ảnh: {input_path}")
        return False

    logger.info(f"Phát hiện {len(results.multi_face_landmarks)} khuôn mặt.")

    all_faces_data: List[Dict[str, Any]] = []
    annotated_image = image_bgr.copy()

    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        face_data = landmarks_to_dict(face_landmarks, w, h)
        face_data["face_index"] = face_idx
        all_faces_data.append(face_data)
        annotated_image = draw_landmarks_on_image(annotated_image, face_landmarks)

    # Lưu JSON
    json_path = output_dir / f"{stem}_landmarks.json"
    save_json(
        {
            "source": str(input_path),
            "image_size": {"width": w, "height": h},
            "faces": all_faces_data,
        },
        json_path,
    )
    logger.info(f"Đã lưu landmarks JSON: {json_path}")

    # Lưu ảnh annotate
    out_img_path = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out_img_path), annotated_image)
    logger.info(f"Đã lưu ảnh annotate: {out_img_path}")

    return True


def process_video(
    input_source: str,
    output_dir: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_num_faces: int = 1,
    is_webcam: bool = False,
) -> bool:
    """
    Xử lý video hoặc webcam: phát hiện landmarks từng frame và lưu MP4 + JSON.

    Args:
        input_source: Đường dẫn video hoặc chỉ số webcam.
        output_dir: Thư mục lưu kết quả.
        min_detection_confidence: Ngưỡng tin cậy detection.
        min_tracking_confidence: Ngưỡng tin cậy tracking.
        max_num_faces: Số mặt tối đa.
        is_webcam: True nếu nguồn là webcam.

    Returns:
        True nếu xử lý thành công.
    """
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(int(input_source) if is_webcam else str(input_source))
    if not cap.isOpened():
        logger.error(f"Không thể mở nguồn video: {input_source}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    source_name = "webcam" if is_webcam else Path(str(input_source)).stem
    out_video_path = output_dir / f"{source_name}_annotated.mp4"
    out_json_path = output_dir / f"{source_name}_landmarks.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    all_frames_data: List[Dict[str, Any]] = []
    frame_count = 0
    faces_detected_count = 0

    logger.info(f"Bắt đầu xử lý {'webcam' if is_webcam else 'video'}: {input_source}")
    if is_webcam:
        logger.info("Nhấn 'q' để dừng webcam.")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            frame_data: Dict[str, Any] = {"frame_index": frame_count, "faces": []}

            if results.multi_face_landmarks:
                faces_detected_count += 1
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    face_data = landmarks_to_dict(face_landmarks, width, height)
                    face_data["face_index"] = face_idx
                    frame_data["faces"].append(face_data)
                    frame = draw_landmarks_on_image(frame, face_landmarks)

            all_frames_data.append(frame_data)
            video_writer.write(frame)
            frame_count += 1

            if total_frames > 0 and frame_count % 30 == 0:
                logger.info(f"  Đã xử lý {frame_count}/{total_frames} frames...")

            if is_webcam:
                cv2.imshow("MediaPipe Face Mesh - DECA_DHMT (q để thoát)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Người dùng dừng webcam.")
                    break

    cap.release()
    video_writer.release()
    if is_webcam:
        cv2.destroyAllWindows()

    # Lưu JSON
    save_json(
        {
            "source": str(input_source),
            "total_frames": frame_count,
            "frames_with_faces": faces_detected_count,
            "video_size": {"width": width, "height": height},
            "fps": fps,
            "frames": all_frames_data,
        },
        out_json_path,
    )

    logger.info(f"Hoàn thành: {frame_count} frames, {faces_detected_count} frames có khuôn mặt.")
    logger.info(f"Video annotate: {out_video_path}")
    logger.info(f"Landmarks JSON: {out_json_path}")
    return True


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir)

    input_str = args.input

    # Kiểm tra xem có phải chỉ số webcam không
    try:
        webcam_index = int(input_str)
        logger.info(f"Chế độ webcam (index={webcam_index})")
        success = process_video(
            input_source=input_str,
            output_dir=output_dir,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            max_num_faces=args.max_num_faces,
            is_webcam=True,
        )
    except ValueError:
        input_path = Path(input_str)
        if not input_path.exists():
            logger.error(f"File không tồn tại: {input_path}")
            sys.exit(1)

        ext = input_path.suffix.lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        if ext in image_exts:
            logger.info(f"Chế độ ảnh tĩnh: {input_path}")
            success = process_image(
                input_path=input_path,
                output_dir=output_dir,
                min_detection_confidence=args.min_detection_confidence,
                max_num_faces=args.max_num_faces,
            )
        elif ext in video_exts:
            logger.info(f"Chế độ video: {input_path}")
            success = process_video(
                input_source=str(input_path),
                output_dir=output_dir,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                max_num_faces=args.max_num_faces,
                is_webcam=False,
            )
        else:
            logger.error(f"Định dạng file không được hỗ trợ: {ext}")
            logger.info(f"Hỗ trợ: ảnh {image_exts}, video {video_exts}, webcam (số nguyên)")
            sys.exit(1)

    if not success:
        logger.error("Xử lý thất bại.")
        sys.exit(1)

    logger.info("✅ Hoàn thành phát hiện landmarks!")


if __name__ == "__main__":
    main()
