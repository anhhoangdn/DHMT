#!/usr/bin/env bash
set -e

echo "[setup.sh] Tạo virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[setup.sh] Cài đặt dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup.sh] Tạo thư mục outputs..."
mkdir -p outputs/{landmarks2d,meshes,textures,renders,videos} reports/{figures,tables} checkpoints

echo ""
echo "✅ Cài đặt hoàn tất!"
echo ""
echo "Bước tiếp theo:"
echo "  1. Thiết lập DECA submodule:"
echo "     git submodule add https://github.com/yfeng95/DECA.git external/DECA"
echo "     git submodule update --init --recursive"
echo "  2. Cài DECA dependencies:"
echo "     pip install -r external/DECA/requirements.txt"
echo "  3. Tải pretrained weights:"
echo "     bash scripts/download_weights.sh"
