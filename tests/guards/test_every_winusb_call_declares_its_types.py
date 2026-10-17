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
cause. ``modules/app_environment.py`` and ``modules/common_utils.py`` fixed
the same defect for their own Win32 calls.

``GetLastError`` called after the foreign call returns reads whatever the
interpreter left in the thread's last error since; a DLL loaded with
``use_last_error=True`` saves it at the call, and ``ctypes.get_last_error()``
reads that copy.

The module imports ``WinDLL`` from ``ctypes``, which has it only on
Windows, so this guard reads its source.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT

_MODULE = 'drivers/winusb_iso.py'
_DLLS = frozenset({'kernel32', 'setupapi', 'winusb'})


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
