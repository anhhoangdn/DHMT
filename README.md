# DHMT — Tái tạo khuôn mặt 3D với DECA & MediaPipe

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Tổng quan đề tài

**DECA_DHMT** là đồ án môn học **Đồ Họa Máy Tính** (DHMT), thực hiện pipeline tái tạo khuôn mặt 3D từ ảnh/video 2D bằng cách kết hợp:

- **MediaPipe Face Mesh** – phát hiện 468 face landmarks 2D/3D tốc độ cao, chạy trên CPU.
- **DECA** (Detailed Expression Capture and Animation) – mô hình học sâu tái tạo hình dạng 3D chi tiết (shape, pose, expression, texture) từ ảnh đơn lẻ, dựa trên FLAME 3D face model.

Đề tài minh họa toàn bộ quy trình từ ảnh đầu vào đến mesh 3D có texture, phục vụ nghiên cứu và demo thực tế.

---

## 🏗️ Kiến trúc / Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                     ĐẦU VÀO (Input)                      │
│         Ảnh (.jpg/.png) | Video (.mp4) | Webcam          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│            STAGE 1 — MediaPipe Face Mesh                 │
│   • Phát hiện khuôn mặt (Face Detection)                 │
│   • Trích xuất 468 landmarks 2D + depth (z)              │
│   • Output: JSON landmarks + ảnh annotate                │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│            STAGE 2 — DECA 3D Reconstruction              │
│   • Encoder CNN (ResNet50) → latent code                 │
│   • FLAME decoder → shape + pose + expression            │
│   • Detail branch → wrinkles, pores                      │
│   • Texture model → UV texture map                       │
│   • Output: .obj mesh + texture + rendered image         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                     ĐẦU RA (Output)                      │
│  outputs/landmarks2d/   — JSON + ảnh annotate            │
│  outputs/meshes/        — file .obj 3D mesh              │
│  outputs/textures/      — UV texture map                 │
│  outputs/renders/       — ảnh rendered 3D                │
│  outputs/videos/        — video annotate (webcam/video)  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
DECA_DHMT/
├── README.md
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── setup.sh                  # Script cài đặt tự động
├── Makefile                  # Make targets
├── LICENSE
├── CONTRIBUTING.md
│
├── configs/
│   ├── default.yaml          # Cấu hình mặc định
│   ├── infer_image.yaml      # Config cho ảnh tĩnh
│   └── infer_video.yaml      # Config cho video/webcam
│
├── src/
│   ├── __init__.py
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── face_detect.py    # Phát hiện khuôn mặt (OpenCV)
│   │   ├── landmark_mediapipe.py  # MediaPipe landmarks ⭐
│   │   └── align.py          # Face alignment
│   ├── recon/
│   │   ├── __init__.py
│   │   ├── run_deca.py       # Wrapper DECA inference ⭐
│   │   ├── fit_3dmm.py       # 3DMM fitting utilities
│   │   └── postprocess.py    # Hậu xử lý mesh
│   ├── render/
│   │   ├── __init__.py
│   │   ├── mesh_export.py    # Xuất .obj mesh
│   │   ├── texture_map.py    # UV texture mapping
│   │   └── visualize.py      # Visualization
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py        # NME, PSNR, 3D error
│   │   └── benchmark.py      # Benchmark pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py             # I/O utilities
│   │   ├── logger.py         # Logger chuẩn
│   │   └── seed.py           # Random seed
│   └── pipeline/
│       ├── __init__.py
│       └── run_pipeline.py   # Pipeline end-to-end ⭐
│
├── scripts/
│   ├── download_weights.sh   # Tải DECA pretrained weights
│   ├── run_image.sh          # Chạy pipeline cho ảnh
│   ├── run_video.sh          # Chạy pipeline cho video
│   └── run_webcam.sh         # Chạy MediaPipe cho webcam
│
├── notebooks/
│   ├── 01_mediapipe_landmarks.ipynb
│   ├── 02_deca_inference.ipynb
│   └── 03_evaluation.ipynb
│
├── docs/
│   ├── pipeline.md
│   ├── dataset.md
│   └── troubleshooting.md
│
├── external/
│   └── DECA/                 # Git submodule (yfeng95/DECA)
│
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   ├── processed/            # Dữ liệu đã xử lý
│   └── samples/              # Ảnh/video mẫu để demo
│
├── outputs/
│   ├── landmarks2d/          # JSON + ảnh landmarks
│   ├── meshes/               # File .obj 3D mesh
│   ├── textures/             # UV texture maps
│   ├── renders/              # Ảnh rendered
│   └── videos/               # Video annotate
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── DECA_DHMT_report_placeholder.txt
│
└── checkpoints/              # Pretrained weights (local)
```

---

## ⚙️ Cài đặt

### Cách 1 — Virtual environment (venv + pip)

```bash
# Clone repo
git clone https://github.com/anhhoangdn/DHMT.git
cd DHMT

# Cài đặt tự động
.\setup.bat

# Hoặc thủ công:
python3 -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
---

## 🔗 Thiết lập DECA Submodule

```bash
# Thêm DECA làm submodule
git submodule add https://github.com/yfeng95/DECA.git external/DECA
git submodule update --init --recursive

# Cài đặt dependencies của DECA
pip install -r external/DECA/requirements.txt
```

---

## 📦 Tải Pretrained Weights DECA

```bash
# Xem hướng dẫn tải weights
bash scripts/download_weights.sh

# Hoặc tải thủ công:
# 1. Truy cập: https://drive.google.com/drive/folders/1h3g4_stMJLJAz_lpRbhCoElDCL5HRfVK
# 2. Tải file: deca_model.tar
# 3. Đặt vào: external/DECA/data/deca_model.tar

# Dùng fetch_data.sh của DECA:
cd external/DECA && bash fetch_data.sh && cd ../..
```

---

## 🚀 Cách chạy

### Landmark Only (MediaPipe)

```bash
# Ảnh tĩnh
python src/preprocess/landmark_mediapipe.py \
    --input data/samples/test.jpg \
    --output_dir outputs/landmarks2d

# Video
python src/preprocess/landmark_mediapipe.py \
    --input data/samples/test.mp4 \
    --output_dir outputs/landmarks2d

# Webcam (index 0)
python src/preprocess/landmark_mediapipe.py \
    --input 0 \
    --output_dir outputs/landmarks2d
# Nhấn 'q' để dừng webcam
```

### DECA Only

```bash
python src/recon/run_deca.py \
    --input data/samples/test.jpg \
    --output outputs \
    --deca-root external/DECA \
    --device cuda
```

### Full Pipeline (Landmark + DECA)

```bash
# Ảnh
python src/pipeline/run_pipeline.py \
    --input data/samples/test.jpg \
    --output outputs \
    --deca-root external/DECA \
    --device cuda

# Video
python src/pipeline/run_pipeline.py \
    --input data/samples/test.mp4 \
    --output outputs \
    --deca-root external/DECA

# Chỉ landmark (bỏ qua DECA)
python src/pipeline/run_pipeline.py \
    --input data/samples/test.jpg \
    --skip-deca

# Webcam (landmark only, DECA không hỗ trợ realtime)
python src/pipeline/run_pipeline.py --input 0 --skip-deca
```

### Dùng Makefile

```bash
make setup                         # Cài đặt môi trường
make landmark INPUT=data/samples/test.jpg
make deca    INPUT=data/samples/test.jpg
make pipeline INPUT=data/samples/test.jpg
make clean                         # Dọn outputs
```

### Dùng shell scripts

```bash
INPUT=data/samples/test.jpg bash scripts/run_image.sh
INPUT=data/samples/test.mp4 bash scripts/run_video.sh
WEBCAM_INDEX=0 bash scripts/run_webcam.sh
```

---

## 📂 Vị trí Output Files

| Loại kết quả | Đường dẫn |
|---|---|
| Landmarks JSON | `outputs/landmarks2d/<name>_landmarks.json` |
| Ảnh annotate landmarks | `outputs/landmarks2d/<name>_annotated.jpg` |
| Video annotate landmarks | `outputs/landmarks2d/<name>_annotated.mp4` |
| 3D Mesh (.obj) | `outputs/deca/<name>/mesh_coarse.obj` |
| UV Texture map | `outputs/deca/<name>/texture.png` |
| Rendered image | `outputs/deca/<name>/rendered_images/<name>.png` |

---

## 🔧 Troubleshooting

### ❌ Thiếu DECA weights

```
⚠️  Không tìm thấy DECA pretrained weights!
```

**Giải pháp:**
```bash
cd external/DECA && bash fetch_data.sh && cd ../..
# Hoặc tải thủ công vào external/DECA/data/deca_model.tar
```

### ❌ CUDA/PyTorch không tương thích

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Giải pháp:**
```bash
# Kiểm tra phiên bản
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi

# Cài lại PyTorch phù hợp từ https://pytorch.org/get-started/locally/
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Hoặc dùng CPU
python src/recon/run_deca.py --device cpu --input ...
```

### ❌ Không tìm thấy script DECA

```
❌ Không tìm thấy script demo của DECA.
```

**Giải pháp:**
```bash
# Kiểm tra submodule
git submodule status
git submodule update --init --recursive

# Kiểm tra nội dung
ls external/DECA/
ls external/DECA/demos/
```

### ❌ ModuleNotFoundError: mediapipe

```bash
pip install mediapipe>=0.10.0
```

### ❌ Webcam không mở được

```bash
# Thử các index khác
python src/preprocess/landmark_mediapipe.py --input 1
# Kiểm tra thiết bị
ls /dev/video*
```

---

## 🎓 Quickstart cho Giảng viên / Demo

```bash
# 1. Clone và cài đặt nhanh
git clone https://github.com/anhhoangdn/DECA_DHMT.git && cd DECA_DHMT
bash setup.sh

# 2. Chạy demo MediaPipe (không cần DECA weights)
python src/preprocess/landmark_mediapipe.py \
    --input data/samples/test.jpg \
    --output_dir outputs/landmarks2d
# → Xem kết quả: outputs/landmarks2d/test_annotated.jpg

# 3. Chạy full pipeline (cần DECA weights)
python src/pipeline/run_pipeline.py \
    --input data/samples/test.jpg \
    --output outputs \
    --device cuda

# 4. Mở notebook demo
jupyter notebook notebooks/01_mediapipe_landmarks.ipynb
```

> **Lưu ý cho giảng viên:** Nếu không có GPU hoặc DECA weights,
> có thể demo chỉ phần MediaPipe bằng `--skip-deca`.

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [DECA — Feng et al., CVPR 2021](https://github.com/yfeng95/DECA)
- [MediaPipe Face Mesh — Google](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [FLAME 3D Face Model](https://flame.is.tue.mpg.de/)
