# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Typed stand-in for the IDS peak SDK (``ids_peak.ids_peak``) in unit tests.

The IDS driver tells what a node is by its type (``isinstance``) and a
failed SDK call by the SDK's exception base, so those are real classes
here; everything else falls back to a MagicMock that keeps its attributes,
as the blanket mock this replaces did, so ``monkeypatch.setattr`` on
``ids_peak.DeviceManager`` and the like still lands on one object.

Modelled on the vendor's own binding, ``ids_peak`` 1.15.0.0.8 (the pin in
``requirements.txt``): ``ids_peak/exceptions.py`` (``Exception`` the base
of every error it raises; ``NotFoundException`` from ``FindNode`` on a name
the map lacks) and ``ids_peak/ids_peak.py`` (the node classes; ``Node``'s
``IsReadable``; ``FloatNode`` and ``IntegerNode``'s ``Unit``;
``EnumerationNode``'s ``CurrentEntry``). The SDK is not installed on the dev
host, so the stand-in is checked against those stubs, not a running SDK.

Install via tests/conftest.py install_mock_deps; never import the real SDK
here.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class Exception(Exception):  # noqa: N818 -- the SDK's own name for its base
    pass


class NotFoundException(Exception):  # noqa: N818 -- mirrors the ids_peak SDK class name exactly
    pass


class BadAccessException(Exception):  # noqa: N818 -- mirrors the ids_peak SDK class name exactly
    pass


class TimeoutException(Exception):  # noqa: N818 -- mirrors the ids_peak SDK class name exactly
    pass


class IOException(Exception):  # noqa: N818 -- mirrors the ids_peak SDK class name exactly
    pass


class InternalErrorException(Exception):  # noqa: N818 -- mirrors the ids_peak SDK class name exactly
    pass


class Node:
    """A node; a test builds one with its value, unit and readability. A value
    that is an exception is raised by the read."""

    def __init__(self, value=None, *, unit: str = '', readable: bool = True):
        self._value = value
        self._unit = unit
        self._readable = readable

    def IsReadable(self) -> bool:
        return self._readable

    def _read(self):
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


class _ValueNode(Node):
    def Value(self):
        return self._read()


class FloatNode(_ValueNode):
    def Unit(self) -> str:
        return self._unit


class IntegerNode(_ValueNode):
    def Unit(self) -> str:
        return self._unit


class BooleanNode(_ValueNode):
    pass


class StringNode(_ValueNode):
    pass


class EnumerationEntryNode(Node):
    def __init__(self, symbolic: str = ''):
        super().__init__()
        self._symbolic = symbolic

    def SymbolicValue(self) -> str:
        return self._symbolic


class EnumerationNode(Node):
    """Its value is its current entry's symbolic name."""

    def CurrentEntry(self) -> EnumerationEntryNode:
        return EnumerationEntryNode(self._read())


class CommandNode(Node):
    pass


class CategoryNode(Node):
    pass


class RegisterNode(Node):
    pass


_TYPED = {
    'Exception': Exception,
    'NotFoundException': NotFoundException,
    'BadAccessException': BadAccessException,
    'TimeoutException': TimeoutException,
    'IOException': IOException,
    'InternalErrorException': InternalErrorException,
    'Node': Node,
    'FloatNode': FloatNode,
    'IntegerNode': IntegerNode,
    'BooleanNode': BooleanNode,
    'StringNode': StringNode,
    'EnumerationNode': EnumerationNode,
    'EnumerationEntryNode': EnumerationEntryNode,
    'CommandNode': CommandNode,
    'CategoryNode': CategoryNode,
    'RegisterNode': RegisterNode,
}

ids_peak = MagicMock(name='ids_peak.ids_peak', **_TYPED)
package = MagicMock(name='ids_peak', ids_peak=ids_peak)
