"""
Root conftest.py — Windows DLL pre-loading and shared configuration.

On Windows, PyTorch uses LoadLibraryExW with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
which requires DLLs to be pre-loaded or their directories registered at the
kernel32 AddDllDirectory level BEFORE torch is imported.

This root conftest pre-loads all torch DLLs using ctypes.CDLL() so they are
cached in the Windows loader before torch's own _load_dll_libraries() runs.

In CI (Linux/macOS), this file has no effect.
"""

import ctypes
import glob
import os
import sys

# ── Windows-only: pre-load torch DLLs before any test imports ─────────────────
if sys.platform == "win32":
    _torch_lib_dir = None

    # Find torch/lib in sys.path
    for _p in sys.path:
        _candidate = os.path.join(_p, "torch", "lib")
        if os.path.isdir(_candidate):
            _torch_lib_dir = _candidate
            break

    if _torch_lib_dir:
        # Register directories
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_torch_lib_dir)
                os.add_dll_directory(os.path.dirname(_torch_lib_dir))
            except OSError:
                pass

        # Pre-load all DLLs using ctypes (standard LoadLibrary, not ExW with restricted flags)
        # This caches them in the Windows DLL loader before torch tries LoadLibraryExW
        _dll_pattern = os.path.join(_torch_lib_dir, "*.dll")
        for _dll in sorted(glob.glob(_dll_pattern)):
            try:
                ctypes.CDLL(_dll)
            except OSError:
                pass  # Some DLLs may fail — that is OK, torch will handle it
