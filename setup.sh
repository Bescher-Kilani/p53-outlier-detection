#!/bin/bash
set -e

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# gmd needs setuptools/wheel/Cython available *before* it builds, with
# build isolation disabled - otherwise it fails with
# "ModuleNotFoundError: No module named 'numpy'/'setuptools'/'Cython'"
# even though they're already installed above.
pip install setuptools wheel Cython
pip install gmd --no-build-isolation

echo "Setup complete. Activate with: source venv/bin/activate"
