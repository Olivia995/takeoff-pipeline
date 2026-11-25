#!/bin/bash
# Double-click this file in Finder to start the takeoff app.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up (takes a minute)..."
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt || { echo "Setup failed."; read -p "Press Return to close."; exit 1; }
fi
.venv/bin/python -m takeoff serve
