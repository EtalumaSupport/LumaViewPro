# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every Win32 function the WinUSB transport calls declares its types, and
its last error is read the way ctypes keeps it.

A foreign function with no ``argtypes`` and ``restype`` passes and returns
C ints: a 64-bit handle -- the device-information set
``SetupDiGetClassDevsW`` returns, an event, a file -- is cut to 32 bits on
the way out and on the way back in, and a check against
``INVALID_HANDLE_VALUE`` can never match. On the 2026-08-26 Windows bench
the stream's device lookup said "WinUSB device not found" on every launch
while the device was present and bound; the truncated handle is the likeliest
cause. ``src/_lumascope/modules/app_environment.py`` and ``src/_lumascope/modules/common_utils.py`` fixed
the same defect for their own Win32 calls.

``GetLastError`` called after the foreign call returns reads whatever the
interpreter left in the thread's last error since; a DLL loaded with
``use_last_error=True`` saves it at the call, and ``ctypes.get_last_error()``
reads that copy.

A declared type converts every argument through its ``from_param``, which
takes a plain Python value or an instance of that type, and refuses an
instance of any other ctypes type. A wrapped argument whose type is not the
declared one raises ``ctypes.ArgumentError`` at the call, on Windows only: on
the 2026-11-11 build PC every Classic's read thread died at its first ISO
read, ``argument 4: TypeError: 'c_bool' object cannot be interpreted as an
integer``, where ``WinUsb_ReadIsochPipeAsap`` declares a ``BOOL``, a C int.

The module imports ``WinDLL`` from ``ctypes``, which has it only on
Windows, so this guard reads its source.
"""

from __future__ import annotations

import ast
import ctypes

from tests.ast_seams import REPO_ROOT, Words

_MODULE = 'src/_lumascope/drivers/winusb_iso.py'
_DLLS = Words({'kernel32', 'setupapi', 'winusb'})


def _tree() -> ast.Module:
    return ast.parse((REPO_ROOT / _MODULE).read_text(encoding='utf-8'))


def _dll_function(node: ast.AST) -> str | None:
    """``kernel32.CreateFileW`` -> ``'kernel32.CreateFileW'``."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in _DLLS
    ):
        return f'{node.value.id}.{node.attr}'
    return None


def test_every_called_win32_function_declares_its_argument_and_return_types():
    tree = _tree()
    called = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _dll_function(node.func))
    }
    # Declared by assignment, or through the module's one helper,
    # ``_declare(function, restype, *argtypes)``, which sets both.
    declared = {'argtypes': set(), 'restype': set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in declared:
                    name = _dll_function(target.value)
                    if name:
                        declared[target.attr].add(name)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_declare'
            and node.args
            and (name := _dll_function(node.args[0]))
        ):
            declared['argtypes'].add(name)
            declared['restype'].add(name)

    assert called, f'no Win32 call found in {_MODULE}: the guard reads nothing'
    assert {
        'argtypes': sorted(called - declared['argtypes']),
        'restype': sorted(called - declared['restype']),
    } == {'argtypes': [], 'restype': []}


def _ctypes_type(name: str, aliases: dict[str, str]) -> type | None:
    """``'_BOOL'`` -> ``ctypes.c_int`` through the module's own aliases;
    None for a name that is not a ctypes simple type (a Structure, a call)."""
    found = getattr(ctypes, aliases.get(name, name), None)
    if isinstance(found, type) and issubclass(found, ctypes._SimpleCData):
        return found
    return None


def test_every_wrapped_argument_is_one_its_declared_type_accepts():
    tree = _tree()
    aliases = {
        node.targets[0].id: node.value.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Name)
    }
    declared = {
        name: [arg.id if isinstance(arg, ast.Name) else None for arg in node.args[2:]]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == '_declare'
        and (name := _dll_function(node.args[0]))
    }

    wrapped = []
    refused = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and (name := _dll_function(node.func))):
            continue
        for position, arg in enumerate(node.args):
            if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)):
                continue
            given = _ctypes_type(arg.func.id, aliases)
            if given is None:
                continue
            wrapped.append(node.lineno)
            types = declared.get(name, [])
            want = types[position] if position < len(types) else None
            want_type = _ctypes_type(want, aliases) if want else None
            try:
                if want_type is None:
                    raise TypeError(f'declared {want!r}, not a ctypes simple type')
                want_type.from_param(given())
            except TypeError as error:
                refused.append(f'line {arg.lineno}: {name} argument {position + 1}: {error}')

    assert wrapped, f'no wrapped Win32 argument found in {_MODULE}: the guard reads nothing'
    assert refused == []


def test_the_last_error_is_the_one_saved_at_the_call():
    tree = _tree()
    reads_get_last_error = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == 'GetLastError'
    )
    loaded_without_saving = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == 'WinDLL')
            or (isinstance(node.func, ast.Name) and node.func.id == 'WinDLL')
        )
        and not any(
            kw.arg == 'use_last_error'
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
    )
    loaded_through_windll = sorted(
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == 'windll'
    )

    assert {
        'GetLastError read': reads_get_last_error,
        'WinDLL without use_last_error': loaded_without_saving,
        'windll': loaded_through_windll,
    } == {'GetLastError read': [], 'WinDLL without use_last_error': [], 'windll': []}
