# External Dependencies — DECA_DHMT

Thư mục này chứa các external dependencies dưới dạng **Git submodule**.

---

## DECA Submodule

**Repository**: https://github.com/yfeng95/DECA  
**Đường dẫn**: `external/DECA/`  
**Mục đích**: 3D Face Reconstruction từ ảnh đơn lẻ

### Thiết lập DECA Submodule

```bash
# Thêm DECA submodule (chỉ cần chạy 1 lần)
git submodule add https://github.com/yfeng95/DECA.git external/DECA

# Cập nhật submodule (mỗi khi clone repo hoặc có update)
git submodule update --init --recursive
```

### Sau khi submodule được khởi tạo

```bash
# Cài đặt dependencies của DECA
pip install -r external/DECA/requirements.txt

# Tải pretrained weights
cd external/DECA && bash fetch_data.sh && cd ../..
# Hoặc:
bash scripts/download_weights.sh
```

### Cấu trúc thư mục DECA (sau khi init)

```
external/DECA/
├── decalib/            # Thư viện core của DECA
│   ├── deca.py
│   ├── models/
│   └── utils/
├── demos/
│   └── demo_reconstruct.py   # Script demo chính
├── data/
│   └── deca_model.tar        # Pretrained weights (cần tải về)
├── requirements.txt
└── README.md
```

---

## Lưu ý

- Thư mục `external/DECA/` được **ignore trong `.gitignore`** vì đây là submodule.
- File `.gitmodules` ở thư mục gốc theo dõi URL và path của submodule.
- Không commit code của DECA vào repo này — chỉ dùng qua submodule reference.

---

## Tài liệu tham khảo

- **Paper**: Feng et al., "Learning an Animatable Detailed 3D Face Model from In-The-Wild Images", CVPR 2021
- **Project page**: https://deca.is.tue.mpg.de/
- **GitHub**: https://github.com/yfeng95/DECA
