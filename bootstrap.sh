
#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "🔹  Env ready – activate anytime with: source .venv/bin/activate"
