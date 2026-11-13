#!/bin/bash
# DEEPFIN app for macOS: double-click to start (the first time: right-click > Open).
# The first start installs DEEPFIN and its libraries into launchers/venv next to this file:
# about 1.5 GB and 5-10 minutes, with an internet connection. Later starts open the app straight away.
cd "$(dirname "$0")" || exit 1
ENV="$PWD/venv"
NEWEST=13  # newest Python 3.x that TensorFlow has builds for: raise it when pypi.org/project/tensorflow/#files lists cp314
GET="from https://www.python.org/downloads/ (TensorFlow does not support newer versions yet)"
RUN=""     # Apple Silicon: always run natively, also from a Terminal opened under Rosetta (TensorFlow has no Intel build there)
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then RUN="arch -arm64"
else NEWEST=12; GET="(the last version TensorFlow supports on Intel Macs): 3.12.10 from https://www.python.org/downloads/release/python-31210/"; fi
if [ ! -x "$ENV/bin/deepfin-gui" ]; then
  MAC=$(sw_vers -productVersion); MIN=12; [ -z "$RUN" ] && MIN=11  # oldest macOS with TensorFlow and the other libraries
  if [ "${MAC%%.*}" -lt "$MIN" ]; then
    echo "DEEPFIN needs macOS $MIN or newer on this Mac, which has macOS $MAC: TensorFlow, which runs the networks,"
    echo "has no build for older versions. Update macOS (Apple menu > Software Update) if this Mac allows it."
    read -r -p "Press Enter to close." _; exit 1
  fi
  PY=""
  for p in python3.12 python3.13 python3.14 python3.15 python3.16 python3; do  # needs 3.12 to 3.$NEWEST with Tkinter (python.org has both)
    if command -v "$p" >/dev/null 2>&1 &&
       $RUN "$p" -c "import sys, tkinter; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, $NEWEST))" 2>/dev/null; then
      PY="$p"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "DEEPFIN needs Python 3.12$([ $NEWEST -gt 12 ] && echo " to 3.$NEWEST") $GET"
    read -r -p "Install it, then double-click this file again. Press Enter to close." _; exit 1
  fi
  SRC=".."  # this repository; or straight from GitHub if this file was downloaded on its own
  [ -f ../pyproject.toml ] || SRC="https://github.com/alexandru-mihai-94/deepfin_clustering/archive/refs/heads/main.zip"
  echo "Installing DEEPFIN with $($RUN "$PY" --version) ($($RUN "$PY" -c 'import platform; print(platform.machine())')), one time only..."
  if ! { $RUN "$PY" -m venv "$ENV" && $RUN "$ENV/bin/python" -m pip install --upgrade pip && $RUN "$ENV/bin/python" -m pip install "$SRC"; }; then
    rm -rf "$ENV"
    read -r -p "Installation failed, see the messages above. Press Enter to close." _; exit 1
  fi
fi
exec $RUN "$ENV/bin/python" -m deepfin.gui
