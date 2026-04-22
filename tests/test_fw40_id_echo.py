# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FW4.0 `id` echo path — release-gate test (docs/FW40_RELEASE_GATE.md §2.1 R3).

Design: docs/FW40_ID_ECHO_LVP_TEST_DESIGN.md (in the Firmware repo).

Invariants verified:
    R3a. Response to a command sent WITH `id` echoes that exact id.
    R3b. Response to a command sent WITHOUT `id` omits id entirely.
    R3c. Events never carry `id`, regardless of whether the enabling
         command carried one.

LED and motor share the framing module on firmware, so every assertion
must hold identically on both boards.

Execution matrix
----------------
Simulator subset (runs on every `pytest` invocation):
    test_response_echoes_id_when_present       — motor + LED
    test_response_echoes_id_over_integer_range — motor + LED
    test_id_echo_on_unknown_cmd_error          — motor + LED
    test_led_and_motor_parity                  — both together
    test_rapid_id_sequence_no_collision        — motor + LED

Hardware subset (gated by --run-hardware, skips per-board if not found):
    test_response_omits_id_when_absent         — raw-line bypass of
                                                 SerialBoard.exchange_json
                                                 (which auto-assigns id)
    test_string_form_command_never_has_id      — exchange_command path
    test_parse_error_preserves_id              — raw malformed JSON input
    test_event_never_carries_id                — motor only; needs real
                                                 EVENTS ON + motion trigger

Out of scope (per the design doc §9): pipelined issuance, cross-session id
collision, `id: null` edge case, host-injected events.
"""
import json
import sys
import time

import pytest

# Heavy deps mocked by tests/conftest.py at collection time.

from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard


hardware = pytest.mark.skipif(
    "--run-hardware" not in sys.argv,
    reason="requires --run-hardware flag and real FW4.0 hardware",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def motor_sim():
    board = SimulatedMotorBoard()
    board.connect()
    yield board
    board.disconnect()


@pytest.fixture(scope='module')
def led_sim():
    board = SimulatedLEDBoard()
    board.connect()
    yield board
    board.disconnect()


@pytest.fixture(scope='module')
def motor_hw():
    """Real MotorBoard on FW4.0. Skipped if not connected or not V4."""
    from drivers.motorboard import MotorBoard
    from drivers.serialboard import ProtocolVersion
    board = MotorBoard()
    if not getattr(board, 'found', False):
        pytest.skip('motor board not found on USB')
    board.connect()
    if board.protocol_version != ProtocolVersion.V4:
        board.disconnect()
        pytest.skip(
            f'motor board running {board.protocol_version.value} '
            '(id-echo test requires FW4.0 / V4)'
        )
    yield board
    board.disconnect()


@pytest.fixture(scope='module')
def led_hw():
    """Real LEDBoard on FW4.0. Skipped if not connected or not V4."""
    from drivers.ledboard import LEDBoard
    from drivers.serialboard import ProtocolVersion
    board = LEDBoard()
    if not getattr(board, 'found', False):
        pytest.skip('LED board not found on USB')
    board.connect()
    if board.protocol_version != ProtocolVersion.V4:
        board.disconnect()
        pytest.skip(
            f'LED board running {board.protocol_version.value} '
            '(id-echo test requires FW4.0 / V4)'
        )
    yield board
    board.disconnect()


# ---------------------------------------------------------------------------
# 4.1 — Response echoes `id` when present (SIM + HW).
# ---------------------------------------------------------------------------

class TestIdEchoPresent:

    def test_motor_sim(self, motor_sim):
        resp = motor_sim.exchange_json({'cmd': 'INFO', 'id': 42})
        assert resp is not None
        assert resp.get('ok') is True
        assert 'id' in resp, 'response missing id field'
        assert resp['id'] == 42

    def test_led_sim(self, led_sim):
        resp = led_sim.exchange_json({'cmd': 'INFO', 'id': 42})
        assert resp is not None
        assert resp.get('ok') is True
        assert 'id' in resp
        assert resp['id'] == 42

    def test_response_echoes_id_over_integer_range_motor_sim(self, motor_sim):
        # Firmware id handling is integer-agnostic — prove it across the
        # 32-bit range and the zero/negative edges.
        for sent_id in (0, 1, 42, 2147483647, -1, -2147483648):
            resp = motor_sim.exchange_json({'cmd': 'INFO', 'id': sent_id})
            assert resp is not None
            assert resp.get('id') == sent_id, (
                f'id echo mismatch: sent {sent_id}, got {resp.get("id")!r}'
            )

    def test_response_echoes_id_over_integer_range_led_sim(self, led_sim):
        for sent_id in (0, 1, 42, 2147483647, -1, -2147483648):
            resp = led_sim.exchange_json({'cmd': 'INFO', 'id': sent_id})
            assert resp is not None
            assert resp.get('id') == sent_id

    @hardware
    def test_motor_hw(self, motor_hw):
        resp = motor_hw.exchange_json({'cmd': 'INFO', 'id': 42})
        assert resp is not None
        assert resp.get('ok') is True
        assert resp.get('id') == 42

    @hardware
    def test_led_hw(self, led_hw):
        resp = led_hw.exchange_json({'cmd': 'INFO', 'id': 42})
        assert resp is not None
        assert resp.get('ok') is True
        assert resp.get('id') == 42


# ---------------------------------------------------------------------------
# 4.2 — Response omits `id` when absent (HW only).
#
# SerialBoard.exchange_json auto-assigns an id for the V4 future-demux path
# (serialboard.py:1019-1022), so we cannot test the "id absent" contract
# through it. Instead, write a raw JSON line directly to the serial port and
# read the response line. Sim has no underlying serial port — hardware-only.
# ---------------------------------------------------------------------------

class TestIdOmittedWhenAbsent:

    @hardware
    def test_motor_hw(self, motor_hw):
        raw = b'{"cmd":"INFO"}\n'
        with motor_hw._lock:
            motor_hw.driver.write(raw)
            line = motor_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is True
        assert 'id' not in resp, (
            f'response carries unexpected id field: {resp.get("id")!r}'
        )

    @hardware
    def test_led_hw(self, led_hw):
        raw = b'{"cmd":"INFO"}\n'
        with led_hw._lock:
            led_hw.driver.write(raw)
            line = led_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is True
        assert 'id' not in resp


# ---------------------------------------------------------------------------
# 4.3 — String-form command never carries id (HW only).
#
# String form (`INFO\n`) cannot carry an id by construction; the response
# must still parse as JSON with no id field.
# ---------------------------------------------------------------------------

class TestStringFormNoId:

    @hardware
    def test_motor_hw(self, motor_hw):
        with motor_hw._lock:
            motor_hw.driver.write(b'INFO\n')
            line = motor_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is True
        assert 'id' not in resp

    @hardware
    def test_led_hw(self, led_hw):
        with led_hw._lock:
            led_hw.driver.write(b'INFO\n')
            line = led_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is True
        assert 'id' not in resp


# ---------------------------------------------------------------------------
# 4.4 — Error response echoes id (SIM + HW).
#
# UNKNOWN_CMD error envelope must carry the request's id so the host demux
# can route the failure back to the right future.
# ---------------------------------------------------------------------------

class TestIdEchoOnError:

    def test_motor_sim(self, motor_sim):
        resp = motor_sim.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 99})
        assert resp is not None
        assert resp.get('ok') is False
        assert resp.get('err') == 'UNKNOWN_CMD'
        assert resp.get('id') == 99

    def test_led_sim(self, led_sim):
        resp = led_sim.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 99})
        assert resp is not None
        assert resp.get('ok') is False
        assert resp.get('err') == 'UNKNOWN_CMD'
        assert resp.get('id') == 99

    @hardware
    def test_motor_hw(self, motor_hw):
        resp = motor_hw.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 99})
        assert resp is not None
        assert resp.get('ok') is False
        assert resp.get('err') == 'UNKNOWN_CMD'
        assert resp.get('id') == 99

    @hardware
    def test_led_hw(self, led_hw):
        resp = led_hw.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 99})
        assert resp is not None
        assert resp.get('ok') is False
        assert resp.get('err') == 'UNKNOWN_CMD'
        assert resp.get('id') == 99


# ---------------------------------------------------------------------------
# 4.5 — Parse-error response preserves id when extractable (HW only).
#
# fw40_framing.handle_line: valid JSON missing 'cmd' → {"ok": false,
# "cmd": "_PARSE", "err": "BAD_PARAM", "id": <echoed if present>}. Sim
# doesn't run firmware parse logic (exchange_json is a dict stub).
# ---------------------------------------------------------------------------

class TestParseErrorPreservesId:

    @hardware
    def test_motor_hw(self, motor_hw):
        raw = b'{"id":77,"foo":"bar"}\n'
        with motor_hw._lock:
            motor_hw.driver.write(raw)
            line = motor_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is False
        assert resp.get('cmd') == '_PARSE'
        assert resp.get('err') == 'BAD_PARAM'
        assert resp.get('id') == 77

    @hardware
    def test_led_hw(self, led_hw):
        raw = b'{"id":77,"foo":"bar"}\n'
        with led_hw._lock:
            led_hw.driver.write(raw)
            line = led_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        assert resp.get('ok') is False
        assert resp.get('cmd') == '_PARSE'
        assert resp.get('err') == 'BAD_PARAM'
        assert resp.get('id') == 77


# ---------------------------------------------------------------------------
# 4.6 — Events never carry id (HW only, motor only).
#
# Simulator's motion-event subsystem doesn't auto-fire on simulated moves
# (2a77950 left that as a TODO); LED firmware events are deferred to FW4.1.
# Motor hardware is the only path that proves the invariant end-to-end.
# ---------------------------------------------------------------------------

class TestEventNoId:

    @hardware
    def test_motor_hw_arrived_event_omits_id(self, motor_hw):
        events = []
        motor_hw.on_event = events.append
        try:
            # Enabling EVENTS carries an id; its response must echo, but
            # the arrived event itself must not.
            resp = motor_hw.exchange_json(
                {'cmd': 'EVENTS', 'mode': 'ON', 'id': 500}
            )
            assert resp is not None
            assert resp.get('id') == 500

            # Trigger a small move so an `arrived` event fires.
            motor_hw.exchange_json(
                {'cmd': 'POS_WRITE', 'axis': 'Z', 'target': 1000, 'id': 501}
            )

            # Poll for the event (~2s max). Real firmware emits within
            # ~100ms of stop-at-target.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if any(e.get('event') == 'arrived' for e in events):
                    break
                time.sleep(0.05)

            arrived = [e for e in events if e.get('event') == 'arrived']
            assert arrived, 'no arrived event received within 2s'
            for ev in arrived:
                assert 'id' not in ev, f'event carried id: {ev!r}'
        finally:
            motor_hw.exchange_json({'cmd': 'EVENTS', 'mode': 'OFF'})
            motor_hw.on_event = None


# ---------------------------------------------------------------------------
# 4.7 — LED and motor parity (SIM + HW).
#
# The release-gate framing-unification claim: both boards share fw40_framing
# so id handling is byte-identical. If they diverge, the unification story
# is wrong and a class of future bugs opens up.
# ---------------------------------------------------------------------------

class TestLedMotorParity:

    def test_sim(self, motor_sim, led_sim):
        # Same id sequence, both boards, identical assertions.
        for sent_id in (111, 222, 2147483647):
            m = motor_sim.exchange_json({'cmd': 'INFO', 'id': sent_id})
            l = led_sim.exchange_json({'cmd': 'INFO', 'id': sent_id})
            assert m.get('id') == sent_id
            assert l.get('id') == sent_id
        # Error path parity.
        m_err = motor_sim.exchange_json({'cmd': 'NOPE', 'id': 333})
        l_err = led_sim.exchange_json({'cmd': 'NOPE', 'id': 333})
        assert m_err.get('ok') is False and m_err.get('id') == 333
        assert l_err.get('ok') is False and l_err.get('id') == 333
        assert m_err.get('err') == l_err.get('err') == 'UNKNOWN_CMD'

    @hardware
    def test_hw(self, motor_hw, led_hw):
        for sent_id in (111, 222, 2147483647):
            m = motor_hw.exchange_json({'cmd': 'INFO', 'id': sent_id})
            l = led_hw.exchange_json({'cmd': 'INFO', 'id': sent_id})
            assert m is not None and l is not None
            assert m.get('id') == sent_id
            assert l.get('id') == sent_id
        m_err = motor_hw.exchange_json({'cmd': 'NOPE', 'id': 333})
        l_err = led_hw.exchange_json({'cmd': 'NOPE', 'id': 333})
        assert m_err.get('ok') is False and m_err.get('id') == 333
        assert l_err.get('ok') is False and l_err.get('id') == 333
        assert m_err.get('err') == l_err.get('err') == 'UNKNOWN_CMD'


# ---------------------------------------------------------------------------
# 4.8 — Rapid id sequence, no collision (SIM + HW).
#
# Exercises the host-side _v4_pending_by_id demux path (SerialBoard.ex-
# change_json:1086-1101) on hardware, and the sim's id-echo on sim.
# ---------------------------------------------------------------------------

class TestRapidIdSequence:

    def test_motor_sim(self, motor_sim):
        for i in range(1, 101):
            resp = motor_sim.exchange_json({'cmd': 'INFO', 'id': i})
            assert resp is not None
            assert resp.get('id') == i

    def test_led_sim(self, led_sim):
        for i in range(1, 101):
            resp = led_sim.exchange_json({'cmd': 'INFO', 'id': i})
            assert resp is not None
            assert resp.get('id') == i

    @hardware
    def test_motor_hw(self, motor_hw):
        for i in range(1, 101):
            resp = motor_hw.exchange_json({'cmd': 'INFO', 'id': i})
            assert resp is not None
            assert resp.get('id') == i

    @hardware
    def test_led_hw(self, led_hw):
        for i in range(1, 101):
            resp = led_hw.exchange_json({'cmd': 'INFO', 'id': i})
            assert resp is not None
            assert resp.get('id') == i
