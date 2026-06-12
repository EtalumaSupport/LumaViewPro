"""Tests for the Phase 1c.3 color-camera boundary-wrapper stubs.

The three stubs (``imread_color`` / ``imwrite_color`` / ``videowriter_color``)
exist so Phase 2 (color-native camera activation) can flip
``camera.is_color_native`` without touching processing modules. Today
they raise ``NotImplementedError`` -- the test confirms the surface is
in place and the right error fires.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.image_utils import imread_color, imwrite_color, videowriter_color


class TestImreadColorStub:
    def test_mono_path_raises_with_message(self, tmp_path):
        path = tmp_path / 'x.tiff'
        with pytest.raises(NotImplementedError, match='mono path'):
            imread_color(path, is_color_native=False)

    def test_color_path_pending(self, tmp_path):
        path = tmp_path / 'x.tiff'
        with pytest.raises(NotImplementedError, match='Phase 2 activation pending'):
            imread_color(path, is_color_native=True)


class TestImwriteColorStub:
    def test_mono_path_raises_with_message(self, tmp_path):
        path = tmp_path / 'x.tiff'
        data = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(NotImplementedError, match='mono fluorescence'):
            imwrite_color(path, data, is_color_native=False)

    def test_color_path_pending(self, tmp_path):
        path = tmp_path / 'x.tiff'
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError, match='Phase 2 activation pending'):
            imwrite_color(path, data, is_color_native=True)


class TestVideoWriterColorStub:
    def test_mono_path_raises_with_message(self):
        with pytest.raises(NotImplementedError, match='mono path'):
            videowriter_color(is_color_native=False)

    def test_color_path_pending(self):
        with pytest.raises(NotImplementedError, match='Phase 2 activation pending'):
            videowriter_color(is_color_native=True)
