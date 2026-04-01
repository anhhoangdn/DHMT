# Troubleshooting — DECA_DHMT

## ❌ Lỗi liên quan đến DECA Weights

### Triệu chứng
```
⚠️  Không tìm thấy DECA pretrained weights!
```
hoặc
```
FileNotFoundError: external/DECA/data/deca_model.tar
```

### Giải pháp

**Cách 1**: Dùng `fetch_data.sh` của DECA (khuyến nghị)
```bash
cd external/DECA
bash fetch_data.sh
cd ../..
```

**Cách 2**: Tải thủ công
```bash
# 1. Truy cập Google Drive
# https://drive.google.com/drive/folders/1h3g4_stMJLJAz_lpRbhCoElDCL5HRfVK
# 2. Tải: deca_model.tar
# 3. Đặt vào:
mkdir -p external/DECA/data
mv ~/Downloads/deca_model.tar external/DECA/data/
```

**Cách 3**: Dùng `gdown`
```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1h3g4_stMJLJAz_lpRbhCoElDCL5HRfVK -O external/DECA/data/
```

**Kiểm tra**:
```bash
ls -la external/DECA/data/
# Phải có: deca_model.tar, generic_model.pkl, ...
```

---

## ❌ CUDA / PyTorch không tương thích

### Triệu chứng
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
hoặc
```
AssertionError: Torch not compiled with CUDA enabled
```
hoặc
```
UserWarning: CUDA initialization: CUDA unknown error
```

### Giải pháp

**Bước 1**: Kiểm tra phiên bản
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi  # Xem driver version và CUDA version
```

**Bước 2**: Cài đúng phiên bản PyTorch
```bash
# Vào https://pytorch.org/get-started/locally/ để chọn đúng
# Ví dụ cho CUDA 11.8:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ví dụ cho CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dùng CPU (không cần GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Bước 3**: Chạy với CPU
```bash
python src/recon/run_deca.py --input data/samples/test.jpg --device cpu
```

---

## ❌ Không tìm thấy Script DECA

### Triệu chứng
```
❌ Không tìm thấy script demo của DECA.
Đã tìm kiếm trong: ['external/DECA/demos/demo_reconstruct.py', ...]
```

### Giải pháp

**Kiểm tra submodule**:
```bash
git submodule status
# Nếu trống hoặc có dấu '-' ở đầu:
git submodule update --init --recursive
```

**Kiểm tra nội dung thư mục**:
```bash
ls external/DECA/
ls external/DECA/demos/ 2>/dev/null || echo "Không có thư mục demos"
```

**Nếu submodule chưa được thêm**:
```bash
git submodule add https://github.com/yfeng95/DECA.git external/DECA
git submodule update --init --recursive
```

---

## ❌ ModuleNotFoundError

### `mediapipe`
```bash
pip install mediapipe>=0.10.0
```

### `cv2`
```bash
pip install opencv-python>=4.8.0
# Nếu bị conflict:
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python>=4.8.0
```

### `src.utils.logger` hoặc các module trong `src/`
```bash
# Chắc chắn chạy từ thư mục gốc của project:
cd /path/to/DECA_DHMT
python src/preprocess/landmark_mediapipe.py --input data/samples/test.jpg
```

---

## ❌ Webcam không mở được

### Triệu chứng
```
Không thể mở nguồn video: 0
```

### Giải pháp
```bash
# Kiểm tra thiết bị webcam
ls /dev/video*

# Thử index khác
python src/preprocess/landmark_mediapipe.py --input 1

# Trên WSL2 (Windows Subsystem for Linux): webcam thường không hoạt động
# Dùng ảnh/video thay thế:
python src/preprocess/landmark_mediapipe.py --input data/samples/test.jpg
```

---

## ❌ Ảnh bị đọc sai màu (BGR/RGB)

### Triệu chứng
Ảnh output có màu sắc kỳ lạ (đỏ thành xanh, xanh thành đỏ).

### Giải pháp
OpenCV đọc ảnh theo BGR, MediaPipe cần RGB. Pipeline đã xử lý tự động,
nhưng nếu bạn xử lý trực tiếp:
```python
image_bgr = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Truyền image_rgb vào MediaPipe
```

---

## ❌ Kết quả landmarks JSON quá lớn

### Vấn đề
File JSON lưu tất cả frame của video dài có thể rất lớn.

### Giải pháp
```python
# Chỉ lưu một số frame nhất định (tùy chỉnh trong code):
# Hoặc nén JSON:
import gzip, json
with gzip.open('landmarks.json.gz', 'wt') as f:
    json.dump(data, f)
```

---

## ❌ Memory / OOM Error khi chạy DECA

### Triệu chứng
```
RuntimeError: CUDA out of memory
```

### Giải pháp
```bash
# Giảm batch size (nếu xử lý nhiều ảnh)
# Hoặc dùng CPU:
python src/recon/run_deca.py --device cpu --input data/samples/test.jpg

# Giải phóng VRAM bằng cách đóng các tiến trình khác
nvidia-smi  # Kiểm tra VRAM usage
```

---

## 💡 Mẹo chung

- **Luôn chạy từ thư mục gốc project** để import đúng module.
- **Kích hoạt virtual environment** trước khi chạy:
  ```bash
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows
  ```
- **Kiểm tra log** để xem thông báo lỗi chi tiết (level INFO/DEBUG).
- Nếu vẫn gặp lỗi, mở **Issue** trên GitHub với log đầy đủ.
