#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "==> Upgrading pip..."
pip install --no-cache-dir --upgrade pip

echo "==> Installing CPU-only PyTorch (lightweight, ~180MB)..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing application dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "==> Build completed successfully!"
