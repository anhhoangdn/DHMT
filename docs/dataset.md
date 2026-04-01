# Dataset và Định dạng Dữ liệu — DECA_DHMT

## Dữ liệu đầu vào được hỗ trợ

### Ảnh tĩnh
| Định dạng | Extension | Ghi chú |
|-----------|-----------|---------|
| JPEG | `.jpg`, `.jpeg` | Khuyến nghị cho ảnh chân dung |
| PNG | `.png` | Hỗ trợ alpha channel |
| BMP | `.bmp` | |
| TIFF | `.tiff` | |
| WebP | `.webp` | |

**Yêu cầu ảnh đầu vào:**
- Khuôn mặt nhìn thẳng hoặc nghiêng nhẹ (≤ 30°)
- Độ phân giải tối thiểu: 224×224 pixel
- Ánh sáng đủ sáng, không bị che khuất nhiều
- Định dạng màu RGB hoặc BGR (OpenCV xử lý tự động)

### Video
| Định dạng | Extension |
|-----------|-----------|
| MP4 | `.mp4` |
| AVI | `.avi` |
| MOV | `.mov` |
| MKV | `.mkv` |
| WebM | `.webm` |
| FLV | `.flv` |

### Webcam
- Chỉ số thiết bị (integer): `0`, `1`, `2`, ...
- Linux: `/dev/video0`, `/dev/video1`, ...

---

## Cấu trúc thư mục dữ liệu

```
data/
├── raw/            # Dữ liệu gốc, chưa xử lý
├── processed/      # Dữ liệu đã tiền xử lý (resize, align)
└── samples/        # Ảnh/video mẫu để demo và test nhanh
    ├── test.jpg    # Ảnh mặc định cho demo
    └── test.mp4    # Video mặc định cho demo
```

---

## Dataset Chuẩn để Đánh giá (Benchmark)

### 300W (Landmark Detection)
- **Mô tả**: 3837 ảnh với 68 landmarks annotated
- **Link**: https://ibug.doc.ic.ac.uk/resources/300-W/
- **Metric**: NME (Normalized Mean Error)

### AFLW (Arbitrary Landmark Localization in the Wild)
- **Mô tả**: Hơn 25,000 ảnh với landmarks trong điều kiện thực tế
- **Link**: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/
- **Metric**: NME, AUC

### VoxCeleb2 (Video)
- **Mô tả**: Video khuôn mặt của 6,112 celebrities
- **Link**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
- **Dùng cho**: Đánh giá temporal consistency

### NoW Benchmark (3D Reconstruction)
- **Mô tả**: Benchmark 3D face reconstruction với scan 3D ground truth
- **Link**: https://ringnet.is.tue.mpg.de/
- **Metric**: Point-to-surface distance (mm)

---

## Định dạng Output

### Landmarks JSON
```json
{
  "source": "data/samples/test.jpg",
  "image_size": {"width": 640, "height": 480},
  "faces": [
    {
      "face_index": 0,
      "num_landmarks": 468,
      "landmarks": [
        {
          "index": 0,
          "x": 0.4823,
          "y": 0.3215,
          "z": -0.0234,
          "x_px": 309,
          "y_px": 154
        }
      ]
    }
  ]
}
```

### DECA Output (per image)
```
outputs/deca/<image_name>/
├── codedict.pkl          # Raw DECA latent codes
├── mesh_coarse.obj       # Coarse 3D mesh (FLAME)
├── mesh_detail.obj       # Detailed 3D mesh (D-DECA)
├── texture.png           # UV texture map (512×512)
└── rendered_images/
    ├── <name>.png        # Rendered face từ góc gốc
    ├── <name>_vis.png    # Visualization với geometry overlay
    └── <name>_shape.png  # Shape-only render
```

---

## Lưu ý về Dữ liệu

- **Không commit** dữ liệu lớn vào git (thêm vào `.gitignore`)
- Sử dụng **Git LFS** nếu cần theo dõi dữ liệu nhỏ (< 100MB)
- Dữ liệu cá nhân: tuân thủ các quy định về bảo mật và quyền riêng tư
- Ảnh trong `data/samples/` chỉ dùng cho mục đích demo/test, không phải dữ liệu training
