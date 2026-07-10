# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The cell-count display edge cannot blit a payload wider than 8 bits.

image_to_texture copies raw bytes as ubyte, so a uint16 array would be read back
as garbage -- a right-aligned 12-bit frame rendered as 2-byte noise. The
depth-aware caller downconverts first (convert_to_8bit at the source's true
depth); the edge fails loud if it ever receives a wide payload, instead of
silently blitting garbage.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# image_utils_kivy imports `from kivy.graphics.texture import Texture`; the test
# env mocks `kivy` but not this submodule. The dtype guard raises before Texture
# is touched, so a permissive mock suffices.
sys.modules.setdefault('kivy.graphics', MagicMock())
sys.modules.setdefault('kivy.graphics.texture', MagicMock())

from ui.image_utils_kivy import image_to_texture


def test_uint16_payload_raises():
    with pytest.raises(ValueError, match='8-bit'):
        image_to_texture(np.zeros((4, 4), dtype=np.uint16))


def test_uint8_payload_clears_guard():
    # 8-bit input passes the dtype guard; the downstream blit is mocked Kivy.
    result = image_to_texture(np.zeros((4, 4), dtype=np.uint8))
    assert result is not None
