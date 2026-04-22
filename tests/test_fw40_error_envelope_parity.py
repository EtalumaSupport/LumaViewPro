# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FW4.0 release-gate §2.1 — error-envelope identical on LED and motor.

Gate language:

    Error envelope is identical on LED and motor:
        {"ok":false, "cmd":<name>, "err":<CODE>, "msg":<text>, "id"?:<id>}
    Error code set shared: BAD_PARAM, NOT_PRESENT, BUSY, HW_ERROR, TIMEOUT,
    SAFETY, UNKNOWN_CMD, RANGE.

Firmware repo has AST-level source tests (tests/test_error_envelope_shape.py)
that prove both handlers build the same dict shape. This test is the
WIRE-level counterpart, on the LVP side: round-trip an error-producing
request, assert the parsed response has the gate shape.

Simulator subset:
    UNKNOWN_CMD — both sim boards emit the error envelope via
    drivers/simulated_{motor,led}board.py:exchange_json (wired 2026-04-22
    commit 4442120 alongside the id-echo test).

Hardware subset (gated by --run-hardware):
    UNKNOWN_CMD + BAD_PARAM (via raw-line parse error — JSON without
    `cmd` hits fw40_framing.handle_line's parse path and returns
    {"ok":false,"cmd":"_PARSE","err":"BAD_PARAM","msg":...}).

Out of scope:
    key-order on the wire (JSON dict ordering is firmware-side and already
    covered by the AST-level test); error codes that only fire under
    specific hardware state (BUSY/TIMEOUT on a moving motor, SAFETY on
    over-current); LED-only codes once they exist.
"""
import json
import sys

import pytest

# Heavy deps mocked by tests/conftest.py at collection time.

from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard


hardware = pytest.mark.skipif(
    "--run-hardware" not in sys.argv,
    reason="requires --run-hardware flag and real FW4.0 hardware",
)


# §2.1 shared error code set. If firmware grows a new error code, add it
# here and the AST-level firmware test simultaneously — both sides of the
# gate must agree.
SHARED_ERR_CODES = frozenset({
    'BAD_PARAM', 'NOT_PRESENT', 'BUSY', 'HW_ERROR',
    'TIMEOUT', 'SAFETY', 'UNKNOWN_CMD', 'RANGE',
})

# Required keys on an error response (id is conditional — see R3).
REQUIRED_ERR_KEYS = frozenset({'ok', 'cmd', 'err', 'msg'})


# ---------------------------------------------------------------------------
# Fixtures — module-scoped, same pattern as test_fw40_id_echo.py
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
    from drivers.motorboard import MotorBoard
    from drivers.serialboard import ProtocolVersion
    board = MotorBoard()
    if not getattr(board, 'found', False):
        pytest.skip('motor board not found on USB')
    board.connect()
    if board.protocol_version != ProtocolVersion.V4:
        board.disconnect()
        pytest.skip(f'motor board running {board.protocol_version.value}')
    yield board
    board.disconnect()


@pytest.fixture(scope='module')
def led_hw():
    from drivers.ledboard import LEDBoard
    from drivers.serialboard import ProtocolVersion
    board = LEDBoard()
    if not getattr(board, 'found', False):
        pytest.skip('LED board not found on USB')
    board.connect()
    if board.protocol_version != ProtocolVersion.V4:
        board.disconnect()
        pytest.skip(f'LED board running {board.protocol_version.value}')
    yield board
    board.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_error_envelope(resp, *, expected_cmd, expected_err, expected_id=None):
    """Assert `resp` is the FW4.0 error envelope shape.

    Release gate §2.1: {"ok":false, "cmd":<name>, "err":<CODE>,
    "msg":<text>, "id"?:<id>}. Error code must be in SHARED_ERR_CODES.
    """
    assert resp is not None, 'expected error response, got None'
    assert isinstance(resp, dict), f'expected dict response, got {type(resp).__name__}'

    missing = REQUIRED_ERR_KEYS - set(resp.keys())
    assert not missing, (
        f'error envelope missing required keys {sorted(missing)}; '
        f'got keys {sorted(resp.keys())}'
    )

    assert resp['ok'] is False, f"expected ok=False, got {resp['ok']!r}"
    assert resp['cmd'] == expected_cmd, (
        f"cmd mismatch: expected {expected_cmd!r}, got {resp['cmd']!r}"
    )
    assert resp['err'] == expected_err, (
        f"err mismatch: expected {expected_err!r}, got {resp['err']!r}"
    )
    assert resp['err'] in SHARED_ERR_CODES, (
        f"err {resp['err']!r} not in release-gate shared set "
        f'{sorted(SHARED_ERR_CODES)}'
    )
    assert isinstance(resp['msg'], str) and resp['msg'], (
        f"msg must be a non-empty string, got {resp['msg']!r}"
    )

    if expected_id is None:
        assert 'id' not in resp, (
            f"response carries unexpected id field: {resp.get('id')!r}"
        )
    else:
        assert resp.get('id') == expected_id, (
            f"id echo mismatch: expected {expected_id}, got {resp.get('id')!r}"
        )


# ---------------------------------------------------------------------------
# UNKNOWN_CMD — sim + HW
# ---------------------------------------------------------------------------

class TestUnknownCmdEnvelope:

    def test_motor_sim(self, motor_sim):
        resp = motor_sim.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 41})
        _assert_error_envelope(
            resp,
            expected_cmd='NOPE_NOT_A_CMD',
            expected_err='UNKNOWN_CMD',
            expected_id=41,
        )

    def test_led_sim(self, led_sim):
        resp = led_sim.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 41})
        _assert_error_envelope(
            resp,
            expected_cmd='NOPE_NOT_A_CMD',
            expected_err='UNKNOWN_CMD',
            expected_id=41,
        )

    def test_motor_and_led_sim_have_identical_keys(self, motor_sim, led_sim):
        """Envelope parity: the set of keys on each board's UNKNOWN_CMD
        response must match exactly. Release gate §2.2 unification."""
        m = motor_sim.exchange_json({'cmd': 'NOPE', 'id': 1})
        l = led_sim.exchange_json({'cmd': 'NOPE', 'id': 1})
        assert set(m.keys()) == set(l.keys()), (
            f'UNKNOWN_CMD envelope key set differs between boards: '
            f'motor={sorted(m.keys())}, led={sorted(l.keys())}'
        )

    @hardware
    def test_motor_hw(self, motor_hw):
        resp = motor_hw.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 41})
        _assert_error_envelope(
            resp,
            expected_cmd='NOPE_NOT_A_CMD',
            expected_err='UNKNOWN_CMD',
            expected_id=41,
        )

    @hardware
    def test_led_hw(self, led_hw):
        resp = led_hw.exchange_json({'cmd': 'NOPE_NOT_A_CMD', 'id': 41})
        _assert_error_envelope(
            resp,
            expected_cmd='NOPE_NOT_A_CMD',
            expected_err='UNKNOWN_CMD',
            expected_id=41,
        )

    @hardware
    def test_motor_and_led_hw_have_identical_keys(self, motor_hw, led_hw):
        m = motor_hw.exchange_json({'cmd': 'NOPE', 'id': 1})
        l = led_hw.exchange_json({'cmd': 'NOPE', 'id': 1})
        assert set(m.keys()) == set(l.keys()), (
            f'UNKNOWN_CMD envelope key set differs between real boards: '
            f'motor={sorted(m.keys())}, led={sorted(l.keys())}'
        )


# ---------------------------------------------------------------------------
# BAD_PARAM parse path — HW only (firmware parse path isn't on sim).
# ---------------------------------------------------------------------------

class TestBadParamEnvelope:
    """Valid JSON without `cmd` → fw40_framing parse path emits
    {"ok":false, "cmd":"_PARSE", "err":"BAD_PARAM", ...}.

    Same wire on both boards; id preserved when extractable.
    """

    @hardware
    def test_motor_hw(self, motor_hw):
        raw = b'{"id":909,"missing_cmd":true}\n'
        with motor_hw._lock:
            motor_hw.driver.write(raw)
            line = motor_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        _assert_error_envelope(
            resp,
            expected_cmd='_PARSE',
            expected_err='BAD_PARAM',
            expected_id=909,
        )

    @hardware
    def test_led_hw(self, led_hw):
        raw = b'{"id":909,"missing_cmd":true}\n'
        with led_hw._lock:
            led_hw.driver.write(raw)
            line = led_hw.driver.readline().decode('utf-8', 'ignore').strip()
        resp = json.loads(line)
        _assert_error_envelope(
            resp,
            expected_cmd='_PARSE',
            expected_err='BAD_PARAM',
            expected_id=909,
        )


# ---------------------------------------------------------------------------
# Shared-code-set audit — meta, keeps this test honest.
# ---------------------------------------------------------------------------

class TestSharedErrorCodeSet:

    def test_shared_codes_match_release_gate(self):
        """SHARED_ERR_CODES in this file must match the release-gate
        doc §2.1 list exactly. If you're adding or renaming an error
        code, update BOTH places in the same commit."""
        gate_set = {
            'BAD_PARAM', 'NOT_PRESENT', 'BUSY', 'HW_ERROR',
            'TIMEOUT', 'SAFETY', 'UNKNOWN_CMD', 'RANGE',
        }
        assert SHARED_ERR_CODES == gate_set, (
            'SHARED_ERR_CODES drifted from docs/FW40_RELEASE_GATE.md §2.1'
        )
