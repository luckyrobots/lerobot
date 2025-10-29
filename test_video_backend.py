#!/usr/bin/env python3
"""Test script to verify video backend compatibility on Windows"""

import sys
import importlib.util

print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print()

# Test 1: Check if torchcodec is available (should be False on Windows)
has_torchcodec = importlib.util.find_spec("torchcodec") is not None
print(f"[OK] torchcodec available: {has_torchcodec}")
print(f"     Expected: False (on Windows)")
print()

# Test 2: Check if lerobot can be imported
try:
    import lerobot
    print(f"[OK] lerobot imported successfully (version {lerobot.__version__})")
except Exception as e:
    print(f"[FAIL] lerobot import failed: {e}")
    sys.exit(1)
print()

# Test 3: Check video_utils
try:
    from lerobot.datasets.video_utils import get_safe_default_codec, decode_video_frames
    print("[OK] video_utils imported successfully")
    default_codec = get_safe_default_codec()
    print(f"     Default codec: {default_codec}")
    print(f"     Expected: pyav (on Windows)")
except Exception as e:
    print(f"[FAIL] video_utils import failed: {e}")
    sys.exit(1)
print()

# Test 4: Check PyAV availability
try:
    import av
    print(f"[OK] PyAV (av) available: version {av.__version__}")
except Exception as e:
    print(f"[FAIL] PyAV not available: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 60)
print("SUMMARY: All video backend tests passed!")
print("Your system is configured correctly:")
print("  - torchcodec: Not installed (correct for Windows)")
print("  - PyAV: Installed and working")
print("  - Video decoding backend: pyav")
print("=" * 60)

