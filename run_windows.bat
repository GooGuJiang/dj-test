@echo off
call conda activate beatthis-auto-dj
if errorlevel 1 (
  echo 无法激活 beatthis-auto-dj 环境。
  echo 请先运行:
  echo   conda env create -f environment.yml
  echo   conda activate beatthis-auto-dj
  echo   python install_beat_this.py
  pause
  exit /b 1
)
python app.py
pause
