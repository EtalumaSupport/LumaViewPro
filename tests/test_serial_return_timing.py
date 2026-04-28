# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Audit F3 regression: ``return_timing=True`` returns ``(response, wire_seconds)``.

The reliability-soak tool depends on this surface to separate wire RTT
from API duration. The tests pin the contract:

* The kwarg is accepted on ``SerialBoard.exchange_command``,
  ``SerialBoard._exchange_command_impl``, ``SerialBoard.exchange_json``,
  ``LEDBoard.exchange_command``, and ``LEDBoard._exchange_v35``.
* On error/early-return paths, callers using the kwarg get
  ``(None, None)`` so tuple-unpacking does not crash.
"""
import inspect
from pathlib import Path


def _read_source(rel_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / rel_path).read_text()


def test_serialboard_exchange_command_accepts_return_timing():
    src = _read_source('drivers/serialboard.py')
    assert 'def exchange_command(self, command, response_numlines=1, timeout=None,\n                         stop_on_empty=False, return_timing=False)' in src
    assert 'def _exchange_command_impl(self, command, response_numlines=1, timeout=None,\n                                stop_on_empty=False, return_timing=False)' in src


def test_ledboard_exchange_command_accepts_return_timing():
    src = _read_source('drivers/ledboard.py')
    assert 'def exchange_command(self, command, response_numlines=1, timeout=None,\n                         stop_on_empty=False, return_timing=False)' in src
    assert 'def _exchange_v35(self, command, timeout=2.0, return_timing=False)' in src


def test_serialboard_returns_tuple_on_early_paths():
    """On early-return paths (silent board, driver=None) the impl must
    return ``(None, None)`` so callers using ``return_timing=True``
    don't crash on tuple-unpack. We assert by source inspection: the
    sentinel ``_none`` is defined and used at every ``return`` inside
    ``_exchange_command_impl``.
    """
    src = _read_source('drivers/serialboard.py')
    impl_start = src.find('def _exchange_command_impl(')
    assert impl_start != -1
    next_def = src.find('\n    def ', impl_start + 1)
    body = src[impl_start:next_def]

    assert '_none = (None, None) if return_timing else None' in body, (
        '_exchange_command_impl must define a return sentinel that adapts '
        'to return_timing; otherwise callers tuple-unpacking on early '
        'returns will crash.'
    )
    # No raw ``return None`` statements should remain in the impl body —
    # they'd break tuple-unpack callers. Whitespace-tolerant.
    forbidden = [line for line in body.splitlines()
                 if line.strip() == 'return None']
    assert not forbidden, (
        f'_exchange_command_impl still contains plain ``return None`` '
        f'on {len(forbidden)} lines — replace with ``return _none``.'
    )


def test_ledboard_v35_returns_tuple_on_early_paths():
    src = _read_source('drivers/ledboard.py')
    impl_start = src.find('def _exchange_v35(')
    assert impl_start != -1
    next_def = src.find('\n    def ', impl_start + 1)
    body = src[impl_start:next_def]

    assert '_none = (None, None) if return_timing else None' in body
    forbidden = [line for line in body.splitlines()
                 if line.strip() == 'return None']
    assert not forbidden, (
        f'_exchange_v35 still contains plain ``return None`` on '
        f'{len(forbidden)} lines — replace with ``return _none``.'
    )


def test_serialboard_exchange_json_accepts_return_timing():
    src = _read_source('drivers/serialboard.py')
    assert 'def exchange_json(self, payload, timeout=None, return_timing=False)' in src


def test_exchange_json_returns_tuple_on_early_paths():
    """exchange_json early-return paths (wrong-protocol, silent board,
    driver=None, timeout, exception) must return ``(None, None)`` when
    ``return_timing=True`` so motor reliability-soak's tuple-unpack
    doesn't crash."""
    src = _read_source('drivers/serialboard.py')
    impl_start = src.find('def exchange_json(')
    assert impl_start != -1
    next_def = src.find('\n    def ', impl_start + 1)
    body = src[impl_start:next_def]

    assert '_none = (None, None) if return_timing else None' in body, (
        'exchange_json must define a return sentinel that adapts to '
        'return_timing; otherwise callers tuple-unpacking on early '
        'returns will crash.'
    )
    forbidden = [line for line in body.splitlines()
                 if line.strip() == 'return None']
    assert not forbidden, (
        f'exchange_json still contains plain ``return None`` on '
        f'{len(forbidden)} lines — replace with ``return _none``.'
    )
