# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The typed pypylon stub makes the Pylon callback layer constructible.

Under the previous blanket MagicMock, ``class ImageHandler(
pylon.ImageEventHandler)`` hit a metaclass conflict at instantiation,
so every test of the callback layer fell back to reading source text.
These tests prove the seams the stub exists to provide; the behavioral
conversions of the former grep-tests build on them.
"""

from unittest.mock import MagicMock

import pytest

from pypylon import genicam, pylon


class TestStubSeams:
    def test_handler_bases_are_real_classes(self):
        assert isinstance(pylon.ImageEventHandler, type)
        assert isinstance(pylon.ConfigurationEventHandler, type)

    def test_exceptions_are_catchable(self):
        with pytest.raises(genicam.RuntimeException):
            raise genicam.RuntimeException('boom')
        assert issubclass(genicam.RuntimeException, Exception)
        assert genicam.RuntimeException is pylon.RuntimeException

    def test_unknown_symbols_fall_back_to_magicmock(self):
        assert isinstance(pylon.TlFactory, MagicMock)

    def test_is_readable_none_node_unreadable(self):
        assert genicam.IsReadable(None) is False
        assert genicam.IsReadable(MagicMock()) is True


class TestDriverHandlersConstructible:
    """The seam the whole F-PYLON conversion family depends on."""

    def _make_parent(self):
        from drivers.pyloncamera import PylonCamera

        parent = PylonCamera.__new__(PylonCamera)
        return parent

    def test_image_handler_instantiates(self):
        from drivers.pyloncamera import ImageHandler

        handler = ImageHandler(self._make_parent())
        assert isinstance(handler, pylon.ImageEventHandler)

    def test_camera_removal_handler_instantiates_and_fires(self):
        from drivers import pyloncamera

        parent = self._make_parent()
        parent._mark_disconnected = MagicMock()
        parent._schedule_async_teardown = MagicMock()
        handler = pyloncamera._CameraRemovalHandler(parent)
        handler.OnCameraDeviceRemoved(camera=MagicMock())
        parent._mark_disconnected.assert_called_once()
        parent._schedule_async_teardown.assert_called_once()
