# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""FW4.0 release-gate §2.1 — capability-probe routing.

Gate language:

    Capability probe returns correct feature array on both boards;
    LVP capability-gating routes correctly on has_feature().

Invariants verified, all headless:

    1. When protocol_version is LEGACY (v3.0.x), V4 methods route to the
       LEGACY path (or no-op) regardless of features[].
    2. When protocol_version is V4 but the baseline feature ('led' for
       LED, 'positions' for motor) is absent, _use_v4() falls back to
       LEGACY — preserves the guard against a partial/stub firmware
       mis-advertising V4.
    3. When protocol_version is V4 + baseline + specific feature (e.g.
       'stim', 'events'), the method routes through exchange_json.
    4. When the specific feature is absent, the method returns None/False
       without touching the serial port.

Board instances constructed via `__new__` — the documented bypass for
unit tests (see MotorBoard._use_v4 docstring: "Defensive getattr() for
tests that construct MotorBoard via __new__"). exchange_json is monkey-
patched on the instance to capture the positive-routing case without
any real serial I/O.

Also sanity-checks simulator default features arrays and the setter.
"""
import pytest

# Heavy deps mocked by tests/conftest.py at collection time.

from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.serialboard import ProtocolVersion
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard


# ---------------------------------------------------------------------------
# Helpers — construct a driver instance without touching hardware.
# ---------------------------------------------------------------------------

def _make_led(protocol_version, features):
    """Build a LEDBoard via __new__ with just the attrs needed for the
    gating code paths. No connect(), no serial. has_feature() reads
    features; _use_v4() reads both."""
    board = LEDBoard.__new__(LEDBoard)
    board.protocol_version = protocol_version
    board.features = list(features)
    return board


def _make_motor(protocol_version, features):
    board = MotorBoard.__new__(MotorBoard)
    board.protocol_version = protocol_version
    board.features = list(features)
    # motion_events_on() calls _install_event_dispatcher() which touches
    # self.on_event. Give it a slot so the attribute access succeeds; the
    # method is idempotent and harmless.
    board.on_event = None
    return board


class _FakeExchange:
    """Capture calls to exchange_json() and return a canned success
    envelope. Assigned to board.exchange_json to sidestep real serial I/O
    while still letting the method under test run to completion."""

    def __init__(self, response=None):
        self.calls = []
        self._response = response

    def __call__(self, payload, timeout=None):
        self.calls.append(payload)
        if self._response is not None:
            return self._response
        resp = {'ok': True, 'cmd': payload.get('cmd')}
        if 'id' in payload:
            resp['id'] = payload['id']
        return resp


# ---------------------------------------------------------------------------
# LEDBoard capability gating
# ---------------------------------------------------------------------------

class TestLedCapabilityGating:

    def test_legacy_blocks_firmware_stim(self):
        board = _make_led(ProtocolVersion.LEGACY, [])
        assert board._use_v4() is False
        assert board.supports_firmware_stim() is False
        assert board.firmware_stim(0, 100, 10, 20, 5) is None

    def test_v4_without_led_baseline_falls_through(self):
        """V4 + features=['stim'] but no 'led' baseline → _use_v4() False.
        Stim-without-led doesn't make physical sense, but the test proves
        the baseline gate fires regardless of other features."""
        board = _make_led(ProtocolVersion.V4, ['stim'])
        assert board._use_v4() is False
        assert board.supports_firmware_stim() is False

    def test_v4_led_baseline_without_stim_blocks_firmware_stim(self):
        board = _make_led(ProtocolVersion.V4, ['led'])
        assert board._use_v4() is True
        assert board.has_feature('stim') is False
        assert board.supports_firmware_stim() is False
        assert board.firmware_stim(0, 100, 10, 20, 5) is None
        assert board.firmware_stim_stop() is None

    def test_v4_with_stim_routes_through_exchange_json(self):
        board = _make_led(ProtocolVersion.V4, ['led', 'stim'])
        assert board.supports_firmware_stim() is True

        fake = _FakeExchange(response={
            'ok': True, 'cmd': 'STIM', 'ch': 0, 'status': 'RUNNING',
        })
        board.exchange_json = fake

        resp = board.firmware_stim(channel=0, mA=100, pulse_ms=10,
                                   period_ms=20, count=5)
        assert resp is not None
        assert resp.get('status') == 'RUNNING'

        assert len(fake.calls) == 1
        sent = fake.calls[0]
        assert sent['cmd'] == 'STIM'
        assert sent['ch'] == 0
        assert sent['mA'] == 100.0
        assert sent['pulse_ms'] == 10.0
        assert sent['period_ms'] == 20.0
        assert sent['count'] == 5


# ---------------------------------------------------------------------------
# MotorBoard capability gating
# ---------------------------------------------------------------------------

class TestMotorCapabilityGating:

    def test_legacy_blocks_motion_events(self):
        board = _make_motor(ProtocolVersion.LEGACY, [])
        assert board._use_v4() is False
        assert board.motion_events_on() is False
        assert board.motion_events_off() is False

    def test_v4_without_positions_baseline_falls_through(self):
        board = _make_motor(ProtocolVersion.V4, ['events'])
        assert board._use_v4() is False
        assert board.motion_events_on() is False

    def test_v4_positions_without_events_blocks_motion_events(self):
        board = _make_motor(ProtocolVersion.V4, ['positions'])
        assert board._use_v4() is True
        assert board.has_feature('events') is False
        assert board.motion_events_on() is False
        assert board.motion_events_off() is False

    def test_v4_with_events_routes_through_exchange_json(self):
        board = _make_motor(ProtocolVersion.V4, ['positions', 'events'])
        fake = _FakeExchange()
        board.exchange_json = fake

        assert board.motion_events_on() is True
        assert board.motion_events_off() is True

        cmds = [c.get('cmd') for c in fake.calls]
        modes = [c.get('mode') for c in fake.calls]
        assert cmds == ['EVENTS', 'EVENTS']
        assert modes == ['ON', 'OFF']

    def test_v4_without_positions_blocks_positions_batch(self):
        board = _make_motor(ProtocolVersion.V4, [])
        assert board.positions_batch() is None


# ---------------------------------------------------------------------------
# Simulator default-features sanity
# ---------------------------------------------------------------------------

class TestSimulatorFeaturesShape:

    def test_motor_default_features(self):
        board = SimulatedMotorBoard()
        feats = board.features
        assert isinstance(feats, list)
        assert all(isinstance(f, str) for f in feats)
        # Baseline is 'positions' — MotorBoard._use_v4 reads it.
        assert 'positions' in feats

    def test_led_default_features(self):
        board = SimulatedLEDBoard()
        feats = board.features
        assert isinstance(feats, list)
        assert all(isinstance(f, str) for f in feats)
        # Baseline is 'led' — LEDBoard._use_v4 reads it.
        assert 'led' in feats

    def test_motor_features_setter_mutates(self):
        board = SimulatedMotorBoard()
        board.features = ['positions']
        assert board.features == ['positions']
        assert board.has_feature('positions') is True
        assert board.has_feature('events') is False

    def test_led_features_setter_mutates(self):
        board = SimulatedLEDBoard()
        board.features = ['led']
        assert board.features == ['led']
        assert board.has_feature('led') is True
        assert board.has_feature('stim') is False

    def test_motor_features_setter_defensive_copy(self):
        """Sim's features setter wraps in list() — mutating the argument
        later must not leak into the board's state."""
        board = SimulatedMotorBoard()
        feats = ['positions', 'events']
        board.features = feats
        feats.append('poison')
        assert 'poison' not in board.features

    def test_led_supports_firmware_stim_gates_on_stim(self):
        """Sim parity with real LEDBoard.supports_firmware_stim — True
        iff 'stim' is in features."""
        board = SimulatedLEDBoard()
        board.features = ['led', 'status']  # no 'stim'
        assert board.supports_firmware_stim() is False
        board.features = ['led', 'stim']
        assert board.supports_firmware_stim() is True
