# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test for SerialBoard.detect_firmware_version JSON-INFO fallback.

Bug (2026-04-24 bench, SN 7162-19): LEDBoard on FW4.0 firmware detected as
LEGACY because detect_firmware_version used only `exchange_command('INFO')`
(plain-text line). LED FW4.0 strictly accepts JSON-framed commands and
returns no response to plain text; motor FW4.0 accepts both, which is why
only LED hit this. Result: _use_v4() returned False post-connect, every V4
path silently fell through to LEGACY paths that don't exist on FW4.0, and
consumers got silent failures (None returns with no user-facing notification).

Fix: when plain-text INFO returns no content, detect_firmware_version falls
back to exchange_json({'cmd':'INFO'}). If that returns a valid dict with
ok=True, drive protocol_version / fw_version / features / firmware_date from
the JSON payload via _apply_json_info.

Invariants verified here:
    D1. Text-INFO success path still works (no regression).
    D2. JSON fallback fires when text returns empty, and sets V4 correctly.
    D3. JSON fallback with non-ok / exception / None response leaves the
        driver in the "non-responsive" terminal state.
    D4. Text-JSON response (body IS JSON on the text path) still routes
        through _parse_json_info as before.
"""
import pytest
from unittest.mock import MagicMock

from drivers.serialboard import SerialBoard, ProtocolVersion


def _make_board_stub():
    """Minimal SerialBoard with just what detect_firmware_version needs."""
    b = SerialBoard.__new__(SerialBoard)
    b._label = '[TEST]'
    b._safe_in_waiting = MagicMock(return_value=0)
    b._detect_response_bytes = 0
    b._parse_legacy_info_text = MagicMock()
    b.driver = MagicMock()
    b.driver.in_waiting = 0
    b.firmware_version = None
    b.firmware_date = None
    b.firmware_responding = False
    b.features = []
    b.protocol_version = ProtocolVersion.LEGACY
    return b


class TestDetectTextPath:
    """D1 — existing text-INFO success path."""

    def test_text_info_v3_legacy_unchanged(self):
        """v3.0.x text INFO → legacy parser runs, no regression."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(
            return_value=['INFO', 'Version 3.0.7', 'Etaluma LED Controller']
        )
        b.detect_firmware_version()
        # Legacy path runs; JSON fallback does NOT fire.
        assert b._parse_legacy_info_text.called
        # exchange_json NOT called — only text path used.
        assert not hasattr(b, 'exchange_json') or \
               not getattr(b.exchange_json, 'called', False)

    def test_text_info_v4_json_body_routes_through_json_parser(self):
        """D4 — some FW4.0 builds emit JSON as text-INFO response body."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=[
            '{"ok":true,"cmd":"INFO","fw_version":"4.0.0","protocol":"4.0",'
            '"features":["id","led","status"],"board":"EL-0940"}'
        ])
        b.detect_firmware_version()
        assert b.protocol_version == ProtocolVersion.V4
        assert b.firmware_version == '4.0.0'
        assert 'led' in b.features
        assert b.firmware_responding is True


class TestDetectJsonFallback:
    """D2 — primary fix: JSON fallback fires when text returns empty."""

    def _info_dict(self, fw='4.0.0', features=None):
        return {
            'ok': True, 'cmd': 'INFO',
            'fw_version': fw, 'fw_date': '2026-04-21',
            'protocol': '4.0',
            'features': features or ['id', 'led', 'status', 'stim'],
            'board': 'EL-0940 Integrated Mainboard',
        }

    def test_empty_text_then_json_ok_classifies_v4(self):
        """The exact scenario from SN 7162-19 bench — LED FW4.0."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        b.exchange_json = MagicMock(return_value=self._info_dict())
        b.detect_firmware_version()
        assert b.protocol_version == ProtocolVersion.V4
        assert b.firmware_version == '4.0.0'
        assert b.firmware_date == '2026-04-21'
        assert 'led' in b.features
        assert 'stim' in b.features
        assert b.firmware_responding is True
        # Fallback was called with the canonical INFO shape.
        b.exchange_json.assert_called_with({'cmd': 'INFO'}, timeout=1.0)

    def test_empty_text_then_empty_list_then_json_ok(self):
        """Text returns empty list — still triggers fallback."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=[])
        b.exchange_json = MagicMock(return_value=self._info_dict())
        b.detect_firmware_version()
        assert b.protocol_version == ProtocolVersion.V4
        assert b.firmware_version == '4.0.0'

    def test_empty_text_then_whitespace_text_then_json_ok(self):
        """Text returns only whitespace — still triggers fallback."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=['  \n', '', '  '])
        b.exchange_json = MagicMock(return_value=self._info_dict())
        b.detect_firmware_version()
        assert b.protocol_version == ProtocolVersion.V4

    def test_v4_via_version_major_when_protocol_field_missing(self):
        """fw_version=4.x classifies as V4 even if 'protocol' key absent."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        info = self._info_dict(fw='4.0.0')
        info.pop('protocol', None)
        b.exchange_json = MagicMock(return_value=info)
        b.detect_firmware_version()
        assert b.protocol_version == ProtocolVersion.V4


class TestDetectBothPathsFail:
    """D3 — both silent → driver correctly reports non-responsive."""

    def test_empty_text_and_empty_json_marks_non_responsive(self):
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        b.exchange_json = MagicMock(return_value=None)
        b.detect_firmware_version()
        assert b.firmware_version is None
        assert b.firmware_responding is False
        assert b.protocol_version == ProtocolVersion.LEGACY

    def test_empty_text_and_err_json_marks_non_responsive(self):
        """JSON returned but ok=False → treat as non-responsive."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        b.exchange_json = MagicMock(return_value={
            'ok': False, 'err': 'UNKNOWN_CMD', 'msg': 'unknown'
        })
        b.detect_firmware_version()
        assert b.firmware_version is None
        assert b.firmware_responding is False
        assert b.protocol_version == ProtocolVersion.LEGACY

    def test_empty_text_and_json_exception_marks_non_responsive(self):
        """JSON fallback raises — driver catches and reports non-responsive.
        Same invariant as the simpler exception branch in detect."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        b.exchange_json = MagicMock(side_effect=RuntimeError('serial closed'))
        b.detect_firmware_version()
        assert b.firmware_version is None
        assert b.firmware_responding is False
        assert b.protocol_version == ProtocolVersion.LEGACY

    def test_empty_text_and_non_dict_json_marks_non_responsive(self):
        """JSON returned non-dict (e.g. str) — defensive handling."""
        b = _make_board_stub()
        b.exchange_command = MagicMock(return_value=None)
        b.exchange_json = MagicMock(return_value="not a dict")
        b.detect_firmware_version()
        assert b.firmware_version is None
        assert b.firmware_responding is False
