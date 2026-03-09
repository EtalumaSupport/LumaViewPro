# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pytest configuration for LumaViewPro tests."""
import sys
import os

# Add project root to path so 'from ledboard import LEDBoard' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
