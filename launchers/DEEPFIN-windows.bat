@echo off
rem DEEPFIN app for Windows: double-click to start.
rem The first start installs DEEPFIN and its libraries into launchers\venv next to this file:
rem about 1.5 GB and 5-10 minutes, with an internet connection. Later starts open the app straight away.
cd /d "%~dp0"
if exist "venv\Scripts\deepfin-gui.exe" goto run

rem needs Python 3.12 or 3.13 with Tkinter, found through the "py" launcher that python.org installs
set "PY="
for %%V in (3.12 3.13) do (
  if not defined PY py -%%V -c "import tkinter" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY (
  echo DEEPFIN needs Python 3.12 ^(or 3.13^) from https://www.python.org/downloads/
  echo Install it, then double-click this file again.
  pause
  exit /b 1
)

rem this repository; or straight from GitHub if this file was downloaded on its own
set "SRC=.."
if not exist "..\pyproject.toml" set "SRC=https://github.com/alexandru-mihai-94/deepfin_clustering/archive/refs/heads/main.zip"
echo Installing DEEPFIN, one time only...
%PY% -m venv venv || goto failed
venv\Scripts\python -m pip install --upgrade pip || goto failed
venv\Scripts\python -m pip install "%SRC%" || goto failed

:run
venv\Scripts\deepfin-gui.exe
exit /b 0

:failed
rmdir /s /q venv 2>nul
echo Installation failed, see the messages above.
pause
exit /b 1
