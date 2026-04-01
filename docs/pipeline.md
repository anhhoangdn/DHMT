# Kiến trúc Pipeline — DECA_DHMT

## Tổng quan

Pipeline DECA_DHMT gồm 2 stage chính:

```
Input → [Stage 1: MediaPipe] → [Stage 2: DECA] → Output
```

---

## Stage 1: MediaPipe Face Mesh

### Mô tả
MediaPipe Face Mesh là framework phát hiện landmarks của Google, chạy theo thời gian thực trên CPU.

### Đặc điểm
- **468 landmarks** bao gồm toạ độ (x, y, z)
- Hoạt động trên CPU, không cần GPU
- Hỗ trợ nhiều khuôn mặt cùng lúc
- Tốc độ: ~30 FPS trên ảnh 720p (CPU)

### Kiến trúc chi tiết
```
Ảnh RGB
    │
    ▼
┌─────────────────────┐
│  BlazeFace Detector │ ← phát hiện vùng khuôn mặt
│  (SSD-based)        │
└──────────┬──────────┘
           │ face ROI
           ▼
┌─────────────────────┐
│  Face Landmark      │ ← predict 468 điểm 3D
│  Model (MobileNet)  │
└──────────┬──────────┘
           │ 468 × (x, y, z)
           ▼
      JSON + annotated image
```

### Output
- `<name>_landmarks.json` — toạ độ 468 landmarks mỗi khuôn mặt
- `<name>_annotated.jpg` — ảnh với landmarks được vẽ

---

## Stage 2: DECA 3D Face Reconstruction

### Mô tả
DECA (Yao Feng et al., CVPR 2021) tái tạo khuôn mặt 3D chi tiết từ một ảnh đơn lẻ,
dựa trên FLAME 3D morphable model.

### Kiến trúc chi tiết
```
Ảnh (224×224)
      │
      ▼
┌─────────────────────┐
│   Encoder           │
│   ResNet-50         │ ← trích xuất đặc trưng ảnh
└──────────┬──────────┘
           │ latent code
           ▼
┌──────────────────────────────────────────┐
│              Coarse Model                │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  │
│  │  Shape  │  │   Pose   │  │ Expr.  │  │
│  │ (100-D) │  │  (6-DOF) │  │ (50-D) │  │
│  └────┬────┘  └────┬─────┘  └───┬────┘  │
│       └────────────┼─────────────┘       │
│                    │                     │
│              FLAME Decoder               │
│          (5023 vertices, triangles)      │
└──────────────────┬───────────────────────┘
                   │ coarse mesh
                   ▼
┌─────────────────────┐
│   Detail Model      │ ← thêm chi tiết (wrinkles, pores)
│   (D-DECA)         │
└──────────┬──────────┘
           │ detailed mesh
           ▼
┌─────────────────────┐
│  Texture Model      │ ← UV texture map từ BFM/FLAME basis
└──────────┬──────────┘
           │
           ▼
     .obj + texture + rendered image
```

### Tham số đầu ra của DECA
| Tham số | Chiều | Mô tả |
|---------|-------|-------|
| `shape` | 100 | Tham số hình dạng FLAME (PCA) |
| `pose` | 6 | Góc đầu (3 global rotation + jaw) |
| `exp` | 50 | Biểu cảm khuôn mặt |
| `tex` | 50 | Tham số texture |
| `cam` | 3 | Weak-perspective camera (scale, tx, ty) |
| `light` | 27 | Spherical harmonics illumination (9×3) |
| `detail` | 128 | Chi tiết da (D-DECA) |

---

## Luồng dữ liệu

```
data/samples/test.jpg
        │
        ├──► [MediaPipe] ──► outputs/landmarks2d/
        │                    ├── test_landmarks.json
        │                    └── test_annotated.jpg
        │
        └──► [DECA] ────────► outputs/deca/test/
                              ├── mesh_coarse.obj
                              ├── mesh_detail.obj
                              ├── texture.png
                              └── rendered_images/
                                  └── test.png
```

---

## Tham khảo

- **DECA paper**: Feng et al., "Learning an Animatable Detailed 3D Face Model from In-The-Wild Images", CVPR 2021
- **FLAME**: Li et al., "Learning a model of facial shape and expression from 4D scans", SIGGRAPH 2017
- **MediaPipe**: Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", arXiv 2019
