#!/usr/bin/env bash
# =============================================================================
# Script tải pretrained weights cho DECA
# =============================================================================
set -e

DECA_ROOT="${DECA_ROOT:-external/DECA}"
CHECKPOINTS_DIR="${DECA_ROOT}/data"

echo "=============================================="
echo " Tải DECA Pretrained Weights"
echo "=============================================="
echo ""
echo "DECA_ROOT: ${DECA_ROOT}"
echo ""

if [ ! -d "${DECA_ROOT}" ]; then
    echo "❌ Không tìm thấy thư mục DECA: ${DECA_ROOT}"
    echo ""
    echo "Thiết lập DECA submodule trước:"
    echo "  git submodule add https://github.com/yfeng95/DECA.git external/DECA"
    echo "  git submodule update --init --recursive"
    exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"

echo "📥 Hướng dẫn tải DECA pretrained weights:"
echo ""
echo "Cách 1 - Tự động (nếu gdown được cài):"
echo "  pip install gdown"
echo "  cd ${DECA_ROOT}"
echo "  bash fetch_data.sh"
echo ""
echo "Cách 2 - Thủ công:"
echo "  1. Truy cập: https://drive.google.com/drive/folders/1h3g4_stMJLJAz_lpRbhCoElDCL5HRfVK"
echo "  2. Tải file deca_model.tar"
echo "  3. Đặt vào: ${CHECKPOINTS_DIR}/deca_model.tar"
echo ""
echo "Cách 3 - Dùng fetch_data.sh của DECA:"
echo "  cd ${DECA_ROOT} && bash fetch_data.sh"
echo ""

# Kiểm tra nếu weights đã tồn tại
if [ -f "${CHECKPOINTS_DIR}/deca_model.tar" ]; then
    echo "✅ Weights đã tồn tại: ${CHECKPOINTS_DIR}/deca_model.tar"
    exit 0
fi

echo "⚠️  Weights chưa được tải. Vui lòng tải theo hướng dẫn trên."
echo ""
echo "Sau khi tải xong, kiểm tra bằng:"
echo "  ls -la ${CHECKPOINTS_DIR}/"
exit 1
