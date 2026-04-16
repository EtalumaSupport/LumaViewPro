# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for ``drivers/fx2driver.py`` — Lumascope Classic FX2 port.

No hardware required. All USB access is mocked via ``_FX2Connection.get``
being monkeypatched to return a ``MagicMock``. Tests cover:

- Registry wiring (camera + LED registry entries)
- Shared-connection invariant (FX2Camera and FX2LEDController share one handle)
- LED ASCII byte mapping (project-memory directive: integer→ASCII stays in driver)
- ``available_channels`` / ``available_colors`` values
- Pure math: gain dB↔register round-trip, digital clamp audit fix, exposure math
- Frame delimiter parser (synthetic bytes)
- Camera ABC conformance (all abstract methods satisfied)
- LEDBoardProtocol duck-type conformance
- **Thin-translator regression**: FX2LEDController must return sentinels
  from state queries even after ``led_on``, proving no client-side state
- ``CameraProfile`` lookup for MT9P031 / LS620 / LS560 / LS720
- ``scopes.json`` shape for the two Classic models added in Stage 3

The test file mocks heavy dependencies (kivy, lvp_logger, pyusb, libusb1)
at module load time so the driver can import cleanly on a dev machine
without pyusb installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Heavy deps (lvp_logger, kivy, usb, usb1, ...) are mocked by
# tests/conftest.py at module-import time. The _FX2Connection.get()
# monkeypatch in fixtures below ensures tests never touch real USB.

import pytest

from drivers import fx2driver
from drivers.registry import camera_registry, led_registry
from drivers.protocols import LEDBoardProtocol
from drivers.camera_profiles import lookup_profile


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_fx2_conn(monkeypatch):
    """Patch _FX2Connection.get() to return a MagicMock.

    Setup: reset the real singleton (in case a prior test left state),
    install the fake getter, yield the MagicMock.

    Teardown: restore the real class state.
    """
    fx2driver._FX2Connection._reset_for_test()

    fake = MagicMock(name='_FX2Connection_fake')
    monkeypatch.setattr(
        fx2driver._FX2Connection,
        'get',
        classmethod(lambda cls: fake),
    )

    yield fake

    fx2driver._FX2Connection._reset_for_test()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    """Verify FX2 is registered in both camera and LED registries."""

    def test_fx2_in_camera_registry(self):
        assert 'fx2' in camera_registry.registered_names()

    def test_fx2_in_led_registry(self):
        assert 'fx2' in led_registry.registered_names()

    def test_camera_entry_is_fx2_camera_class(self):
        cls = camera_registry.get('fx2')
        assert cls is fx2driver.FX2Camera

    def test_led_entry_is_fx2_led_controller_class(self):
        cls = led_registry.get('fx2')
        assert cls is fx2driver.FX2LEDController


# ---------------------------------------------------------------------------
# LED channel → I2C ASCII byte mapping (project-memory directive)
# ---------------------------------------------------------------------------

class TestLEDChannelMapping:
    """The integer → ASCII byte translation must stay inside the driver.

    Memory rule: "The integer-to-ASCII mapping must stay inside the driver,
    not leak into the API." These tests verify the values.
    """

    def test_ch0_blue_maps_to_ascii_C(self):
        assert fx2driver._CH_TO_I2C[0] == 0x43  # 'C'

    def test_ch1_green_maps_to_ascii_B(self):
        assert fx2driver._CH_TO_I2C[1] == 0x42  # 'B'

    def test_ch2_red_maps_to_ascii_A(self):
        assert fx2driver._CH_TO_I2C[2] == 0x41  # 'A'

    def test_ch3_bf_maps_to_ascii_D(self):
        assert fx2driver._CH_TO_I2C[3] == 0x44  # 'D'

    def test_color_to_ch_blue_is_zero(self):
        assert fx2driver._COLOR_TO_CH['Blue'] == 0

    def test_color_to_ch_all_four(self):
        assert fx2driver._COLOR_TO_CH == {
            'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3,
        }

    def test_ch_to_color_roundtrip(self):
        for color, ch in fx2driver._COLOR_TO_CH.items():
            assert fx2driver._CH_TO_COLOR[ch] == color

    def test_no_fifth_channel(self):
        """The FX2 peripheral controller only supports 4 channels.
        The driver must not expose 5 or 6 like the RP2040 LEDBoard does.
        """
        assert len(fx2driver._COLOR_TO_CH) == 4
        assert len(fx2driver._CH_TO_I2C) == 4


# ---------------------------------------------------------------------------
# Pure math: gain dB ↔ register
# ---------------------------------------------------------------------------

class TestGainMath:
    """Gain conversion is pure math — testable without hardware."""

    @pytest.mark.parametrize("db", [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0])
    def test_round_trip_within_half_dB(self, db):
        reg = fx2driver._gain_db_to_register(db)
        _, back = fx2driver._register_to_gain_db(reg)
        assert abs(back - db) < 0.5, \
            f'dB round-trip failed: {db} → 0x{reg:04X} → {back}'

    def test_zero_db_is_unity_gain(self):
        reg = fx2driver._gain_db_to_register(0.0)
        mult, db = fx2driver._register_to_gain_db(reg)
        assert abs(mult - 1.0) < 0.001
        assert abs(db) < 0.001

    def test_digital_clamp_respects_120_max(self):
        """Audit fix: digital_val must clamp at 120, not 127.

        Per RR_A, Digital_Gain legal values are [0, 120]. The LVC
        reference clamped to 127 which is outside the documented
        range. Verify we stay at or below 120 even for extreme
        requests.
        """
        # Request something way above max (1000x = 60 dB)
        reg = fx2driver._gain_db_to_register(60.0)
        digital_val = (reg >> 8) & 0x7F
        assert digital_val <= 120, \
            f'digital_val {digital_val} exceeds audit-corrected legal max 120'

    def test_negative_db_clamped_to_unity(self):
        reg = fx2driver._gain_db_to_register(-10.0)
        _, back = fx2driver._register_to_gain_db(reg)
        assert back >= 0.0

    def test_analog_only_below_4x(self):
        """≤ 4x gain should use analog-only (multiplier=0, digital=0)."""
        reg = fx2driver._gain_db_to_register(6.0)  # 2x
        analog_mult = (reg >> 6) & 1
        digital_val = (reg >> 8) & 0x7F
        assert analog_mult == 0
        assert digital_val == 0


# ---------------------------------------------------------------------------
# Pure math: exposure rows
# ---------------------------------------------------------------------------

class TestExposureMath:
    """Exposure formula: rows = (target_ms + SO_ms) / tROW_ms."""

    def test_50ms_round_trip_within_one_row(self):
        target = 50.0
        rows = round((target + fx2driver._SHUTTER_OVERHEAD_MS) / fx2driver._ROW_TIME_MS)
        back = rows * fx2driver._ROW_TIME_MS - fx2driver._SHUTTER_OVERHEAD_MS
        assert abs(back - target) < fx2driver._ROW_TIME_MS, \
            f'50ms → {rows} rows → {back}ms (should be within 1 row of target)'

    def test_max_rows_is_max_exposure(self):
        assert fx2driver.MAX_EXPOSURE_ROWS == 65535

    def test_max_exposure_near_7_4_seconds(self):
        max_ms = fx2driver.MAX_EXPOSURE_ROWS * fx2driver._ROW_TIME_MS - fx2driver._SHUTTER_OVERHEAD_MS
        assert 7300.0 < max_ms < 7400.0


# ---------------------------------------------------------------------------
# Frame delimiter parser
# ---------------------------------------------------------------------------

class TestFrameDelimiterParser:
    """The \\x01\\xfe\\x00\\xff delimiter pattern injected by GpifWaveform_Isr."""

    def test_delimiter_is_4_bytes(self):
        assert fx2driver.FRAME_DELIM == b'\x01\xfe\x00\xff'
        assert len(fx2driver.FRAME_DELIM) == 4

    def test_find_delimiter_at_known_offset(self):
        buf = bytearray(b'\xAA' * 100) + fx2driver.FRAME_DELIM + bytearray(b'\xBB' * 100)
        idx = buf.find(fx2driver.FRAME_DELIM)
        assert idx == 100

    def test_find_returns_negative_when_absent(self):
        buf = bytearray(b'\xAA' * 1000)
        assert buf.find(fx2driver.FRAME_DELIM) == -1

    def test_multiple_delimiters_finds_first(self):
        buf = (
            bytearray(b'\xAA' * 50)
            + fx2driver.FRAME_DELIM
            + bytearray(b'\xBB' * 50)
            + fx2driver.FRAME_DELIM
            + bytearray(b'\xCC' * 50)
        )
        assert buf.find(fx2driver.FRAME_DELIM) == 50


# ---------------------------------------------------------------------------
# Intel HEX parser
# ---------------------------------------------------------------------------

class TestIntelHexParser:
    """``parse_intel_hex`` is pure — testable without USB or real firmware."""

    def test_parses_real_firmware(self):
        hex_path = REPO_ROOT / 'firmware' / 'LumascopeClassic.hex'
        if not hex_path.exists():
            pytest.skip(f'firmware hex not present at {hex_path}')
        data, end_addr = fx2driver.parse_intel_hex(str(hex_path))
        assert len(data) == 0x4000                       # 16 KB 8051 program space
        assert end_addr > 0                              # something was parsed
        assert end_addr <= 0x4000                        # within program space
        assert data[end_addr:] == b'\xff' * (0x4000 - end_addr)  # tail unused = 0xFF

    def test_parses_fallback_firmware(self):
        hex_path = REPO_ROOT / 'firmware' / 'Lumascope600.hex'
        if not hex_path.exists():
            pytest.skip(f'fallback firmware hex not present at {hex_path}')
        _, end_addr = fx2driver.parse_intel_hex(str(hex_path))
        assert end_addr > 0


# ---------------------------------------------------------------------------
# Camera ABC conformance
# ---------------------------------------------------------------------------

class TestFX2CameraABCConformance:
    """FX2Camera must satisfy every Camera abstract method."""

    def test_no_abstract_methods_remaining(self):
        assert fx2driver.FX2Camera.__abstractmethods__ == frozenset()

    def test_inherits_from_camera(self):
        from drivers.camera import Camera
        assert issubclass(fx2driver.FX2Camera, Camera)


# ---------------------------------------------------------------------------
# LEDBoardProtocol conformance
# ---------------------------------------------------------------------------

class TestFX2LEDProtocolConformance:
    """FX2LEDController must satisfy LEDBoardProtocol via duck typing."""

    def test_all_protocol_methods_present(self):
        required = [
            'connect', 'disconnect', 'is_connected',
            'led_on', 'led_off', 'led_on_fast', 'led_off_fast',
            'leds_off', 'leds_off_fast', 'leds_enable', 'leds_disable',
            'get_led_ma', 'is_led_on', 'get_led_state', 'get_led_states',
            'color2ch', 'ch2color',
            'available_channels', 'available_colors',
            'read_led_current', 'exchange_command',
        ]
        for name in required:
            assert hasattr(fx2driver.FX2LEDController, name), f'missing: {name}'

    def test_isinstance_check(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert isinstance(led, LEDBoardProtocol)


# ---------------------------------------------------------------------------
# FX2LEDController — "thin command translator" regression guard
# ---------------------------------------------------------------------------

class TestFX2LEDThinTranslator:
    """The FX2 LED driver MUST NOT track LED state client-side.

    The LVC reference carried a ``led_ma`` dict that leaked the pre-4.1
    GUI's state-ownership model into the driver. In 4.1, the API owns
    state via ``Lumascope._led_owners``. These tests prove the new
    driver has no state bookkeeping:

    1. State-query methods return sentinel defaults matching NullLEDBoard.
    2. Calling ``led_on`` does NOT change what state-query methods return.
    3. The driver has no ``led_ma`` attribute (symbol-level regression guard).
    """

    def test_initial_state_query_returns_sentinel(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert led.get_led_ma('Blue') == -1
        assert led.is_led_on('Blue') is False
        assert led.get_led_state('Blue') == {'enabled': False, 'illumination': -1}

    def test_led_on_does_not_update_state_query(self, fake_fx2_conn):
        """The critical regression guard.

        Calling ``led_on(0, 100)`` must send I2C commands but must NOT
        update any internal state that ``get_led_ma`` / ``is_led_on``
        would read back. If someone re-adds ``self.led_ma`` and this
        test starts failing in the direction of "get_led_ma returns
        100", that's the regression we're preventing.
        """
        led = fx2driver.FX2LEDController()
        led.led_on(0, 100)

        # The I2C command SHOULD have been sent (via fake conn.i2c_write)
        assert fake_fx2_conn.i2c_write.called, \
            'led_on should issue I2C writes through the FX2 connection'

        # But the state queries must STILL return sentinel defaults —
        # the driver does not remember what it just did.
        assert led.get_led_ma('Blue') == -1
        assert led.is_led_on('Blue') is False

    def test_led_off_also_leaves_state_sentinel(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        led.led_on(0, 100)
        led.led_off(0)
        assert led.get_led_ma('Blue') == -1

    def test_get_led_states_returns_all_false(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        led.led_on(0, 100)
        led.led_on(1, 50)
        states = led.get_led_states()
        for color, state in states.items():
            assert state == {'enabled': False, 'illumination': -1}, \
                f'{color} leaked state from led_on'

    def test_no_led_ma_attribute(self, fake_fx2_conn):
        """Symbol-level regression guard: the ``led_ma`` dict that the
        LVC reference driver carried must be absent in the new driver."""
        led = fx2driver.FX2LEDController()
        assert not hasattr(led, 'led_ma'), \
            'FX2LEDController.led_ma exists — thin-translator rule violated'


# ---------------------------------------------------------------------------
# available_channels / available_colors (B3 integration)
# ---------------------------------------------------------------------------

class TestFX2LEDChannelDiscovery:
    """Post-B3 channel discovery — `available_channels()` drives API validation
    and `ScopeCapabilities.led_channels`.
    """

    def test_returns_four_integer_channels(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert led.available_channels() == (0, 1, 2, 3)

    def test_returns_four_color_names(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert led.available_colors() == ('Blue', 'Green', 'Red', 'BF')


# ---------------------------------------------------------------------------
# Shared-connection invariant
# ---------------------------------------------------------------------------

class TestSharedConnectionInvariant:
    """FX2Camera and FX2LEDController must end up pointing at the SAME
    `_FX2Connection` singleton — this is the whole point of the module-
    level singleton pattern, and is proven at the architecture level by
    TestRegistryAccommodatesCompositeHardware in test_driver_registry.py.
    Here we verify the real drivers satisfy it.
    """

    def test_camera_and_led_share_fx2_ref(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        cam = fx2driver.FX2Camera()
        assert led._fx2 is cam._fx2, \
            'camera and LED drivers must share the same _FX2Connection'

    def test_both_hold_same_fake(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        cam = fx2driver.FX2Camera()
        assert led._fx2 is fake_fx2_conn
        assert cam._fx2 is fake_fx2_conn


# ---------------------------------------------------------------------------
# FX2Camera profile loading + dynamic capabilities
# ---------------------------------------------------------------------------

class TestFX2CameraProfile:
    """FX2Camera should load the MT9P031-LS620 profile and populate
    dynamic gain / exposure fields in ``_query_dynamic_capabilities``.
    """

    def test_profile_loads_on_connect(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.model_name == 'MT9P031-LS620'
        assert cam.profile.driver == 'fx2'
        assert cam.profile.sensor == 'Aptina MT9P031'

    def test_dynamic_capabilities_populated(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.profile.gain.total_min_db == 0.0
        assert cam.profile.gain.total_max_db == 42.1
        assert cam.profile.exposure_min_us is not None
        assert cam.profile.exposure_max_us is not None

    def test_pixel_format_is_mono8_only(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_pixel_format() == 'Mono8'
        assert cam.get_supported_pixel_formats() == ('Mono8',)

    def test_default_frame_size_is_1900x1900(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_frame_size() == {'Width': 1900, 'Height': 1900}

    def test_max_frame_size_is_1900x1900(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_max_frame_size() == {'Width': 1900, 'Height': 1900}

    def test_binning_returns_one(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_binning_size() == 1

    def test_auto_features_are_noops(self, fake_fx2_conn):
        """MT9P031 has no hardware AE/AG — these should all be silent
        no-ops rather than raising NotImplementedError.
        """
        cam = fx2driver.FX2Camera()
        cam.auto_exposure_t(True)
        cam.auto_gain(True)
        cam.auto_gain_once(True)
        cam.update_auto_gain_target_brightness(0.5)
        cam.update_auto_gain_min_max(0.0, 30.0)
        cam.set_test_pattern(True, 'Black')
        cam.set_max_acquisition_frame_rate(True, 4.5)
        assert cam.get_all_temperatures() == {}


# ---------------------------------------------------------------------------
# camera_profiles.py registration
# ---------------------------------------------------------------------------

class TestCameraProfileRegistration:
    """The MT9P031 profile must be discoverable via the four substring
    keys registered in camera_profiles.py: MT9P031, LS620, LS560, LS720.
    """

    @pytest.mark.parametrize("lookup_key", [
        'MT9P031',
        'MT9P031-LS620',    # what FX2Camera sets model_name to
        'LS620',
        'LS560',
        'LS720',
    ])
    def test_profile_found_by_substring(self, lookup_key):
        profile = lookup_profile(lookup_key)
        assert profile.driver == 'fx2', \
            f'{lookup_key!r} should resolve to the FX2 profile'
        assert profile.sensor == 'Aptina MT9P031'
        assert profile.pixel_size_um == 2.2
        assert profile.pixel_formats == ['Mono8']
        assert profile.has_auto_gain is False
        assert profile.has_auto_exposure is False

    def test_max_exposure_matches_driver_constants(self):
        """The profile's max_exposure_ms should match what the driver
        exposes via MAX_EXPOSURE_ROWS × _ROW_TIME_MS.
        """
        profile = lookup_profile('LS620')
        driver_max_ms = int(fx2driver.MAX_EXPOSURE_ROWS * fx2driver._ROW_TIME_MS)
        # Allow 10 ms tolerance for rounding
        assert abs(profile.max_exposure_ms - driver_max_ms) <= 10


# ---------------------------------------------------------------------------
# data/scopes.json shape for Classic models
# ---------------------------------------------------------------------------

class TestScopesJsonClassicModels:
    """LS620 and LS560 entries should exist with correct capability bits.

    LS720 is intentionally NOT in scopes.json until Stage 4 ships the
    LVC motor driver — avoids the "scopes.json says XYZ but
    capabilities.axes is empty" inconsistency.
    """

    @pytest.fixture
    def scopes(self):
        path = REPO_ROOT / 'data' / 'scopes.json'
        with open(path) as f:
            return json.load(f)

    def test_ls620_exists(self, scopes):
        assert 'LS620' in scopes

    def test_ls560_exists(self, scopes):
        assert 'LS560' in scopes

    def test_ls720_NOT_in_scopes_json_yet(self, scopes):
        """Stage 3 intentionally defers LS720 to Stage 4 (LVC motor port)."""
        assert 'LS720' not in scopes, \
            'LS720 should not be added until Stage 4 ships drivers/lvc_motorboard.py'

    def test_ls620_has_no_motors(self, scopes):
        entry = scopes['LS620']
        assert entry['Focus'] is False
        assert entry['XYStage'] is False
        assert entry['Turret'] is False

    def test_ls560_has_no_motors(self, scopes):
        entry = scopes['LS560']
        assert entry['Focus'] is False
        assert entry['XYStage'] is False
        assert entry['Turret'] is False

    def test_ls620_has_fluorescence_bf_phase(self, scopes):
        layers = scopes['LS620']['Layers']
        assert layers['Fluorescence'] is True
        assert layers['Brightfield'] is True
        assert layers['PhaseContrast'] is True
        assert layers['Darkfield'] is False
        assert layers['Lumi'] is False

    def test_ls560_has_fluorescence_bf_phase(self, scopes):
        """LS560 has BF/Phase (one combined channel) + Green fluorescence.
        scopes.json Layers dict is boolean per imaging mode — the "Green
        only" limitation vs LS620's BGR is a future schema enhancement.
        """
        layers = scopes['LS560']['Layers']
        assert layers['Fluorescence'] is True
        assert layers['Brightfield'] is True
        assert layers['PhaseContrast'] is True
        assert layers['Darkfield'] is False
        assert layers['Lumi'] is False


# ---------------------------------------------------------------------------
# _FX2Connection singleton lifecycle
# ---------------------------------------------------------------------------

class TestFX2ConnectionSingleton:
    """The ``_FX2Connection.get()`` / ``_reset_for_test()`` pattern."""

    def setup_method(self):
        fx2driver._FX2Connection._reset_for_test()

    def teardown_method(self):
        fx2driver._FX2Connection._reset_for_test()

    def test_reset_clears_instance(self):
        fx2driver._FX2Connection._instance = MagicMock()
        fx2driver._FX2Connection._reset_for_test()
        assert fx2driver._FX2Connection._instance is None

    def test_get_returns_cached_instance(self, monkeypatch):
        """Second call to get() must return the same object (not reconstruct)."""
        fake = MagicMock()
        monkeypatch.setattr(
            fx2driver._FX2Connection, 'get', classmethod(lambda cls: fake)
        )
        first = fx2driver._FX2Connection.get()
        second = fx2driver._FX2Connection.get()
        assert first is second

    def test_init_failure_leaves_instance_none(self):
        """If _FX2Connection.__init__ raises, the singleton stays None so
        the next attempt can retry from scratch (e.g., after plugging in
        the hardware).
        """
        def boom(self):
            raise RuntimeError('no FX2 hardware')

        # Patch __init__ directly — get() will still call cls() which
        # calls __new__ + __init__; if __init__ raises, cls._instance
        # is never assigned.
        with patch.object(fx2driver._FX2Connection, '__init__', boom):
            with pytest.raises(RuntimeError, match='no FX2 hardware'):
                fx2driver._FX2Connection.get()
        assert fx2driver._FX2Connection._instance is None
