"""
Xuất 3D mesh sang định dạng .obj.
TODO: Hỗ trợ thêm .ply, .glb.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    uvs: Optional[np.ndarray] = None,
    uv_faces: Optional[np.ndarray] = None,
    mtl_name: Optional[str] = None,
) -> None:
    """
    Xuất mesh ra file .obj.

    Args:
        vertices: Mảng vertices (N, 3).
        faces: Mảng faces (F, 3), 0-indexed.
        output_path: Đường dẫn file .obj.
        uvs: UV coordinates (M, 2), tùy chọn.
        uv_faces: UV face indices (F, 3), tùy chọn.
        mtl_name: Tên material file, tùy chọn.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# DECA_DHMT - Exported OBJ\n")
        if mtl_name:
            f.write(f"mtllib {mtl_name}.mtl\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # UV coordinates
        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Faces (1-indexed trong OBJ format)
        if uvs is not None and uv_faces is not None:
            for fi, face in enumerate(faces):
                uv_face = uv_faces[fi]
                f.write(
                    f"f {face[0]+1}/{uv_face[0]+1}"
                    f" {face[1]+1}/{uv_face[1]+1}"
                    f" {face[2]+1}/{uv_face[2]+1}\n"
                )
        else:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    logger.info(f"Đã xuất mesh OBJ: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")


def load_obj(obj_path: Path):
    """
    Tải mesh từ file .obj (đơn giản).

    Args:
        obj_path: Đường dẫn file .obj.

    Returns:
        Tuple (vertices, faces) as numpy arrays.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Xử lý f v/vt/vn format
                indices = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(indices)

    logger.info(f"Đã tải OBJ: {obj_path} ({len(vertices)} vertices, {len(faces)} faces)")
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
