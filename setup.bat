@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

echo ================================================
echo  DECA_DHMT -- Cai dat moi truong (Windows)
echo ================================================
echo.

:: ── 0. Kiem tra Conda co trong PATH khong ────────
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Khong tim thay 'conda' trong PATH.
    echo         Cai Miniconda hoac Anaconda truoc:
    echo         https://docs.conda.io/en/latest/miniconda.html
    pause & exit /b 1
)
echo [OK] Tim thay conda.
for /f "tokens=*" %%v in ('conda --version 2^>^&1') do echo [OK] %%v
echo.

:: ── 1. Tao moi truong chinh: deca_dhmt ───────────
echo [1/4] Tao conda environment: deca_dhmt
echo       (MediaPipe, OpenCV, numpy moi...)

conda env list | findstr /C:"deca_dhmt" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] deca_dhmt da ton tai -- cap nhat...
    conda env update -n deca_dhmt -f envs\environment_main.yml --prune
) else (
    conda env create -f envs\environment_main.yml
)
if errorlevel 1 (
    echo [ERROR] Tao deca_dhmt that bai.
    pause & exit /b 1
)
echo [OK] deca_dhmt san sang.
echo.

:: ── 2. Tao moi truong DECA: deca_env ─────────────
echo [2/4] Tao conda environment: deca_env
echo       (torch==1.8, numpy==1.22, DECA dependencies...)

conda env list | findstr /C:"deca_env" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] deca_env da ton tai -- cap nhat...
    conda env update -n deca_env -f envs\environment_deca.yml --prune
) else (
    conda env create -f envs\environment_deca.yml
)
if errorlevel 1 (
    echo [ERROR] Tao deca_env that bai.
    pause & exit /b 1
)
echo [OK] deca_env san sang.
echo.

:: ── 3. Cai DECA dependencies vao deca_env ────────
echo [3/4] Cai DECA dependencies vao deca_env...
echo       (Bo qua requirements.txt goc vi chua version cu khong tuong thich)

conda run -n deca_env pip install git+https://github.com/mattloper/chumpy
if errorlevel 1 (
    echo [WARNING] Cai chumpy that bai -- kiem tra ket noi mang.
)

conda run -n deca_env pip install scikit-image==0.19.3 PyYAML==5.4.1 face-alignment==1.3.5 yacs==0.1.8 kornia==0.5.11 ninja fvcore
if errorlevel 1 (
    echo [ERROR] Cai DECA dependencies that bai.
    pause & exit /b 1
)
echo [OK] DECA dependencies da cai.
echo.

:: ── 4. Cai render tools vao deca_dhmt ────────────
echo [4/4] Cai render tools (trimesh, pyrender, imageio)...

conda run -n deca_dhmt pip install trimesh pyrender "imageio[ffmpeg]"
if errorlevel 1 (
    echo [WARNING] Cai render tools that bai.
    echo           Chay thu cong: pip install trimesh pyrender imageio[ffmpeg]
) else (
    echo [OK] Render tools da cai.
)
echo.

:: ── Tao thu muc output ────────────────────────────
echo Tao cac thu muc can thiet...
for %%d in (
    outputs\landmarks2d
    outputs\meshes
    outputs\textures
    outputs\renders
    outputs\videos
    reports\figures
    reports\tables
    checkpoints
    data\raw
    data\processed
    data\samples
) do (
    if not exist %%d (
        mkdir %%d
        echo   [+] %%d
    )
)
echo.

:: ── Kiem tra submodule DECA ───────────────────────
if not exist external\DECA\.git (
    echo [WARNING] DECA submodule chua duoc init.
    echo.
    echo Chay cac lenh sau de hoan tat cai dat:
    echo.
    echo   git submodule update --init --recursive
    echo   scripts\download_weights.bat
) else (
    echo [OK] DECA submodule da co tai: external\DECA
)
echo.

:: ── Tong ket ─────────────────────────────────────
echo ================================================
echo  Cai dat hoan tat!
echo ================================================
echo.
echo Cac moi truong da tao:
echo   deca_dhmt  -- chay MediaPipe + render (Stage 1)
echo   deca_env   -- chay DECA              (Stage 2)
echo.
echo Buoc tiep theo:
echo.
echo   1. Kich hoat moi truong chinh:
echo      conda activate deca_dhmt
echo.
echo   2. Chay pipeline:
echo      python src\pipeline\run_pipeline.py --input data\samples\face.jpg --output outputs --deca_root external\DECA --device cpu
echo      python src\pipeline\run_pipeline.py --input data\samples\test.mp4 --output outputs --deca_root external\DECA --device cpu
echo      python src\pipeline\run_pipeline.py --input 0 --output outputs --deca_root external\DECA --device cpu
echo.
echo   3. Render video 360:
echo      python scripts\render_360.py --obj outputs\deca\test\test\test_detail.obj --output outputs\videos\test_360.mp4 --frames 72
echo.
echo   4. Chi chay Stage 1 (MediaPipe):
echo      python src\pipeline\run_pipeline.py --input data\samples\face.jpg --skip_deca
echo.
echo ================================================
echo.
pause
