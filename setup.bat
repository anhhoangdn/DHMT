@echo off
setlocal

:: Check Python version (require 3.10 or 3.11)
python - <<PY
import sys
v = sys.version_info
if not (v.major == 3 and v.minor in (10,11)):
    print(f"[setup.bat] ERROR: Python {v.major}.{v.minor} detected. Please use Python 3.10 or 3.11.")
    sys.exit(1)
print(f"[setup.bat] Python {v.major}.{v.minor} OK")
PY
if errorlevel 1 exit /b 1

echo [setup.bat] Tao virtual environment...
python -m venv venv || exit /b 1

echo [setup.bat] Kich hoat virtual environment...
call venv\Scripts\activate || exit /b 1

echo [setup.bat] Cai dat dependencies...
python -m pip install --upgrade pip || exit /b 1
pip install -r requirements.txt || exit /b 1

echo [setup.bat] Tao thu muc outputs...
if not exist outputs\landmarks2d mkdir outputs\landmarks2d
if not exist outputs\meshes      mkdir outputs\meshes
if not exist outputs\textures    mkdir outputs\textures
if not exist outputs\renders     mkdir outputs\renders
if not exist outputs\videos      mkdir outputs\videos

if not exist reports\figures mkdir reports\figures
if not exist reports\tables  mkdir reports\tables

if not exist checkpoints mkdir checkpoints

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
