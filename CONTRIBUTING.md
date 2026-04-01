# Hướng dẫn Đóng góp (Contributing Guide)

Cảm ơn bạn đã quan tâm đến dự án **DECA_DHMT**! Dưới đây là quy trình đóng góp.

---

## 📋 Quy trình Fork / Branch / PR

### 1. Fork repository

Nhấn nút **Fork** trên GitHub để tạo bản sao về tài khoản của bạn.

### 2. Clone về máy

```bash
git clone https://github.com/<your-username>/DECA_DHMT.git
cd DECA_DHMT
```

### 3. Tạo branch mới

Đặt tên branch theo quy ước:
- `feature/<tên-tính-năng>` — thêm tính năng mới
- `fix/<tên-lỗi>` — sửa lỗi
- `docs/<tên-tài-liệu>` — cập nhật tài liệu
- `refactor/<tên-module>` — cải thiện code

```bash
git checkout -b feature/them-tinh-nang-xyz
```

### 4. Cài đặt môi trường phát triển

```bash
bash setup.sh
source venv/bin/activate
```

### 5. Thực hiện thay đổi

- Viết code theo chuẩn **PEP 8**.
- Thêm **type hints** cho tất cả hàm.
- Viết **docstring** bằng tiếng Việt hoặc tiếng Anh (Google style).
- Giữ commit nhỏ, rõ ràng.

### 6. Commit thay đổi

```bash
git add .
git commit -m "feat: thêm tính năng xyz"
```

Quy ước commit message (Conventional Commits):
- `feat:` — tính năng mới
- `fix:` — sửa lỗi
- `docs:` — tài liệu
- `refactor:` — cải thiện code không thêm tính năng
- `chore:` — cập nhật cấu hình, dependencies

### 7. Push và tạo Pull Request

```bash
git push origin feature/them-tinh-nang-xyz
```

Sau đó vào GitHub, tạo Pull Request từ branch của bạn vào `main`.

---

## ✅ Checklist trước khi tạo PR

- [ ] Code chạy không có lỗi
- [ ] Có docstring cho hàm/class mới
- [ ] Đã kiểm tra với ảnh mẫu trong `data/samples/`
- [ ] Không commit weights, dữ liệu lớn, hay file `.env`
- [ ] Mô tả PR rõ ràng: làm gì, tại sao

---

## 🐛 Báo cáo lỗi (Bug Report)

Mở **Issue** trên GitHub với thông tin:
1. Mô tả lỗi
2. Các bước tái tạo lỗi
3. Log/thông báo lỗi đầy đủ
4. Hệ điều hành, phiên bản Python, GPU (nếu có)

---

## 💡 Đề xuất tính năng

Mở **Issue** với label `enhancement` và mô tả:
1. Vấn đề cần giải quyết
2. Giải pháp đề xuất
3. Tại sao tính năng này hữu ích

---

Mọi thắc mắc, liên hệ qua Issues hoặc email nhóm. Cảm ơn!
