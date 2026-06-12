# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Typed stand-in for the pypylon SDK in unit tests.

The previous blanket ``sys.modules`` MagicMock made every ``pylon.*``
symbol a MagicMock, so ``ImageHandler`` (which subclasses
``pylon.ImageEventHandler``) could not be instantiated -- a metaclass
conflict at class construction -- and the whole Pylon callback layer
was only testable by reading source text. This stub provides real
classes for the seams that need them (event-handler bases to subclass
and instantiate; exception types usable in ``except`` clauses; int
enum constants) while module-level ``__getattr__`` falls back to
MagicMock for everything else, so untouched call sites behave exactly
as they did under the blanket mock.

Install via tests/conftest.py install_mock_deps; never import the real
pypylon here -- the stub must work on machines without the SDK.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Exceptions -- real BaseException subclasses so production `except`
# clauses work when tests raise them through driver code.
# ---------------------------------------------------------------------------


class GenericException(Exception):  # noqa: N818 -- mirrors the pypylon SDK class name exactly
    pass


class RuntimeException(GenericException):
    pass


class TimeoutException(GenericException):
    pass


class AccessException(GenericException):
    pass


class LogicalErrorException(GenericException):
    pass


# ---------------------------------------------------------------------------
# Subclassable handler bases -- mirror the pypylon virtual-method
# surface the drivers override. No-op bodies: the SDK base does nothing
# until the subclass overrides.
# ---------------------------------------------------------------------------


class ImageEventHandler:
    """Stand-in for pylon.ImageEventHandler (C++ CImageEventHandler)."""

    def OnImageGrabbed(self, camera, grab_result) -> None:
        pass

    def OnImagesSkipped(self, camera, count_of_skipped_images) -> None:
        pass


class ConfigurationEventHandler:
    """Stand-in for pylon.ConfigurationEventHandler."""

    def OnCameraDeviceRemoved(self, camera) -> None:
        pass

    def OnAttach(self, camera) -> None:
        pass

    def OnOpened(self, camera) -> None:
        pass

    def OnClosed(self, camera) -> None:
        pass


class AcquireContinuousConfiguration:
    """Stand-in for the SDK's default continuous-grab configuration."""


def IsReadable(node) -> bool:
    """genicam.IsReadable stand-in: a None node is unreadable; anything
    a test supplies (MagicMock or fake node) counts as readable."""
    return node is not None


def _stub_module(name: str, members: dict) -> types.ModuleType:
    """Build a module whose unknown attributes fall back to MagicMock,
    mirroring the blanket-mock behavior for symbols the stub does not
    model explicitly."""
    mod = types.ModuleType(name)
    for key, value in members.items():
        setattr(mod, key, value)

    def _fallback(attr, _name=name):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return MagicMock(name=f'{_name}.{attr}')

    mod.__getattr__ = _fallback
    return mod


_PYLON_MEMBERS = {
    'ImageEventHandler': ImageEventHandler,
    'ConfigurationEventHandler': ConfigurationEventHandler,
    'AcquireContinuousConfiguration': AcquireContinuousConfiguration,
    'GenericException': GenericException,
    'RuntimeException': RuntimeException,
    'TimeoutException': TimeoutException,
    'AccessException': AccessException,
    'LogicalErrorException': LogicalErrorException,
    # Enum stand-ins: distinct ints so identity/equality checks behave.
    'GrabStrategy_OneByOne': 0,
    'GrabStrategy_LatestImageOnly': 1,
    'GrabLoop_ProvidedByInstantCamera': 2,
    'RegistrationMode_Append': 3,
    'RegistrationMode_ReplaceAll': 4,
    'Cleanup_Delete': 5,
}

_GENICAM_MEMBERS = {
    'GenericException': GenericException,
    'RuntimeException': RuntimeException,
    'TimeoutException': TimeoutException,
    'AccessException': AccessException,
    'LogicalErrorException': LogicalErrorException,
    'IsReadable': IsReadable,
}

pylon = _stub_module('pypylon.pylon', _PYLON_MEMBERS)
genicam = _stub_module('pypylon.genicam', _GENICAM_MEMBERS)
pypylon = _stub_module('pypylon', {'pylon': pylon, 'genicam': genicam})
