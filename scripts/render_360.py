"""
Render video xoay 360° từ file .obj của DECA.
Chạy trong môi trường deca_dhmt.

Sử dụng:
    python scripts/render_360.py --obj outputs/deca/test/test/test.obj --output outputs/videos/test_360.mp4
"""
import argparse
import numpy as np
import imageio
from pathlib import Path

def render_360(obj_path: str, output_path: str, n_frames: int = 72, size: int = 512):
    try:
        import trimesh
        import pyrender
    except ImportError:
        print("Cài thêm: pip install trimesh pyrender imageio[ffmpeg]")
        return

    print(f"Đọc mesh: {obj_path}")
     # Thành này — load mesh, bỏ texture, dùng màu xám
    mesh = trimesh.load(obj_path, force='mesh')
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[200, 200, 200,255])
    # mesh = trimesh.load(obj_path, process=False)
    # if hasattr(mesh, 'geometry'):  # nếu load ra Scene thì lấy mesh đầu tiên
    #     mesh = list(mesh.geometry.values())[0]
    # Căn giữa mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    mesh.vertices /= scale

    scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = scene.add(render_mesh)

    # Lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    light2 = pyrender.DirectionalLight(color=np.ones(3)*0.5, intensity=1.5)
    l2_pose = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=float)
    scene.add(light2, pose=l2_pose)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi/4.0)
    cam_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]], dtype=float)
    scene.add(camera, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(size, size)

    frames = []
    print(f"Render {n_frames} frames...")
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        # Xoay mesh quanh trục Y
        rot = np.array([
            [ np.cos(angle), 0, np.sin(angle), 0],
            [0,              1, 0,              0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0,              0, 0,              1],
        ])
        scene.set_pose(mesh_node, pose=rot)
        color, _ = renderer.render(scene)
        frames.append(color)

        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n_frames} frames")

    renderer.delete()

    # Lưu video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=24)
    print(f"✅ Video lưu tại: {output_path}")

    # Lưu thêm GIF
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames[::2], fps=12)  # bỏ bớt frame cho GIF nhẹ hơn
    print(f"✅ GIF lưu tại: {gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj",    required=True,  help="Đường dẫn file .obj")
    parser.add_argument("--output", default="outputs/videos/render_360.mp4")
    parser.add_argument("--frames", type=int, default=72,  help="Số frames (72 = 5 giây 24fps)")
    parser.add_argument("--size",   type=int, default=512, help="Kích thước ảnh render")
    args = parser.parse_args()

    render_360(
        obj_path    = args.obj,
        output_path = args.output,
        n_frames    = args.frames,
        size        = args.size,
    )

if __name__ == "__main__":
    main()