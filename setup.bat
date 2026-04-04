@echo off
setlocal

echo [setup.bat] Tao virtual environment...
python -m venv venv

echo [setup.bat] Kich hoat virtual environment...
call venv\Scripts\activate

echo [setup.bat] Cai dat dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [setup.bat] Tao thu muc outputs...
mkdir outputs\landmarks2d
mkdir outputs\meshes
mkdir outputs\textures
mkdir outputs\renders
mkdir outputs\videos

mkdir reports\figures
mkdir reports\tables

mkdir checkpoints

echo.
echo ✅ Cai dat hoan tat!
echo.
echo Buoc tiep theo:
echo 1. Thiet lap DECA submodule:
echo    git submodule add https://github.com/yfeng95/DECA.git external/DECA
echo    git submodule update --init --recursive
echo.
echo 2. Cai DECA dependencies:
echo    pip install -r external/DECA/requirements.txt
echo.
echo 3. Tai pretrained weights:
echo    bash scripts/download_weights.sh

pause
