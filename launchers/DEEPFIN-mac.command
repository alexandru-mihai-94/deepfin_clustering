#!/bin/bash
# DEEPFIN app for macOS: double-click to start (the first time: right-click > Open).
# The first start installs DEEPFIN and its libraries into launchers/venv next to this file:
# about 1.5 GB and 5-10 minutes, with an internet connection. Later starts open the app straight away.
cd "$(dirname "$0")" || exit 1
ENV="$PWD/venv"
if [ ! -x "$ENV/bin/deepfin-gui" ]; then
  PY=""
  for p in python3.12 python3.13 python3; do  # needs 3.12 or 3.13 with Tkinter (python.org has both)
    if command -v "$p" >/dev/null 2>&1 &&
       "$p" -c 'import sys, tkinter; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; then
      PY="$p"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "DEEPFIN needs Python 3.12 (or 3.13) from https://www.python.org/downloads/"
    read -r -p "Install it, then double-click this file again. Press Enter to close." _; exit 1
  fi
  SRC=".."  # this repository; or straight from GitHub if this file was downloaded on its own
  [ -f ../pyproject.toml ] || SRC="https://github.com/alexandru-mihai-94/deepfin_clustering/archive/refs/heads/main.zip"
  echo "Installing DEEPFIN with $("$PY" --version), one time only..."
  if ! { "$PY" -m venv "$ENV" && "$ENV/bin/python" -m pip install --upgrade pip && "$ENV/bin/python" -m pip install "$SRC"; }; then
    rm -rf "$ENV"
    read -r -p "Installation failed, see the messages above. Press Enter to close." _; exit 1
  fi
fi
exec "$ENV/bin/deepfin-gui"
