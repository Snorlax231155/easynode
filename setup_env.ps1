# setup_env.ps1 - Create Python 3.11 venv and install all required packages

$PY311 = "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\python.exe"
$VENV  = ".\venv_capsnet"

if (-Not (Test-Path $PY311)) {
    Write-Host "ERROR: Python 3.11 not found at $PY311" -ForegroundColor Red
    Write-Host "Please run the installer first from: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    exit 1
}

Write-Host "Creating virtual environment with Python 3.11..." -ForegroundColor Cyan
& $PY311 -m venv $VENV

Write-Host "Activating and installing packages..." -ForegroundColor Cyan
& "$VENV\Scripts\pip.exe" install --upgrade pip setuptools wheel

# Core ML stack
& "$VENV\Scripts\pip.exe" install tensorflow==2.15.0
& "$VENV\Scripts\pip.exe" install tensorflow-io==0.36.0
& "$VENV\Scripts\pip.exe" install tensorflow-io-gcs-filesystem==0.36.0

# Data / preprocessing
& "$VENV\Scripts\pip.exe" install numpy scikit-learn scikit-image tqdm
& "$VENV\Scripts\pip.exe" install Pillow opencv-python
& "$VENV\Scripts\pip.exe" install matplotlib pandas PyWavelets

# Game theory
& "$VENV\Scripts\pip.exe" install nashpy

# Scipy (for hausdorff)
& "$VENV\Scripts\pip.exe" install scipy

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Activate with: .\venv_capsnet\Scripts\Activate.ps1"
Write-Host "Then train  with: python train_capsnet.py"
