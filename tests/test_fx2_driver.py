# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for ``drivers/fx2driver.py`` -- Lumascope Classic FX2 port.

No hardware required. All USB access is mocked via ``_FX2Connection.get``
being monkeypatched to return a ``MagicMock``. Tests cover:

- Registry wiring (camera + LED registry entries)
- Shared-connection invariant (FX2Camera and FX2LEDController share one handle)
- LED ASCII byte mapping (project-memory directive: integer->ASCII stays in driver)
- ``available_channels`` values
- Pure math: gain dB<->register round-trip, digital clamp audit fix, exposure math
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
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Heavy deps (lvp_logger, kivy, usb, usb1, ...) are mocked by
# tests/conftest.py at module-import time. The _FX2Connection.get()
# monkeypatch in fixtures below ensures tests never touch real USB.

import pytest

from drivers import fx2driver
from drivers.registry import DriverRegistry, NoLiveDriverError, camera_registry, led_registry
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

    Also neutralizes the streaming-start paths. ``FX2Camera.__init__``
    calls ``connect()`` -> ``start_grabbing()`` -> the connection's
    ``start_stream``, which, under a MagicMock'd libusb1/winusb, records ~16 xfer
    callbacks holding a bound-method ref back to the camera. The
    MagicMock's call-args store makes that a cycle the GC can't break
    (~210 MB leaked per FX2Camera() instance). Tests in this file
    never exercise the streaming loop, so a no-op patch is safe here.
    Any future test that DOES want to exercise streaming should use a
    dedicated fixture that wires up a real-enough stream source.
    """
    fx2driver._FX2Connection._reset_for_test()

    fake = MagicMock(name='_FX2Connection_fake')
    monkeypatch.setattr(
        fx2driver._FX2Connection,
        'get',
        classmethod(lambda cls: fake),
    )
    monkeypatch.setattr(fx2driver.FX2Camera, 'start_grabbing', lambda self: None)
    monkeypatch.setattr(fx2driver.FX2Camera, 'stop_grabbing', lambda self: None)

    yield fake

    fx2driver._FX2Connection._reset_for_test()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not fx2driver._FX2_GATE.available,
    reason='FX2 prerequisites (pyusb / libusb1) not installed -- drivers '
    'intentionally not registered (issue #635)',
)
class TestRegistryWiring:
    """Verify FX2 is registered in both camera and LED registries.

    Skipped on systems missing pyusb (or libusb1 on macOS/Linux): per #635
    the FX2 drivers refuse to register on installs where they cannot work,
    so the registration assertion is meaningless there.
    """

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


class TestRegistryGatedOnPrereqs:
    """Issue #635 regression: FX2 drivers must NOT register when pyusb/libusb1
    are unavailable. Otherwise the registry attempts instantiation, raises
    ImportError, and dumps a confusing traceback on every no-LVC install."""

    def test_an_unavailable_fx2_is_declined_with_its_reason(self, monkeypatch):
        # Not registered, so the registry never tries it; declined, so a
        # camera that is not found says which driver did not load and why.
        monkeypatch.setattr(
            fx2driver,
            '_FX2_GATE',
            fx2driver._FX2_GATE._replace(
                available=False, unavailable_reason='pyusb is not installed'
            ),
        )
        registry = DriverRegistry('camera')

        class Dummy:
            pass

        assert fx2driver._register_if_fx2_available(registry, 'fx2', priority=80)(Dummy) is Dummy
        assert registry.registered_names() == []
        with pytest.raises(NoLiveDriverError) as caught:
            registry.create('auto')
        assert caught.value.fallback.declined == ('fx2: pyusb is not installed',)


# ---------------------------------------------------------------------------
# LED channel -> I2C ASCII byte mapping (project-memory directive)
# ---------------------------------------------------------------------------


class TestLEDChannelMapping:
    """The integer -> ASCII byte translation must stay inside the driver.

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

    def test_no_fifth_channel(self):
        """The FX2 peripheral controller only supports 4 channels.
        The driver must not expose more like the EL-0940 LEDBoard does.
        """
        assert len(fx2driver._CH_TO_I2C) == 4


# ---------------------------------------------------------------------------
# Pure math: gain dB <-> register
# ---------------------------------------------------------------------------


class TestGainMath:
    """Gain conversion is pure math -- testable without hardware."""

    @pytest.mark.parametrize('db', [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0])
    def test_round_trip_within_half_dB(self, db):
        reg = fx2driver._gain_db_to_register(db)
        _, back = fx2driver._register_to_gain_db(reg)
        assert abs(back - db) < 0.5, f'dB round-trip failed: {db} -> 0x{reg:04X} -> {back}'

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
        assert digital_val <= 120, (
            f'digital_val {digital_val} exceeds audit-corrected legal max 120'
        )

    def test_negative_db_clamped_to_unity(self):
        reg = fx2driver._gain_db_to_register(-10.0)
        _, back = fx2driver._register_to_gain_db(reg)
        assert back >= 0.0

    def test_analog_only_below_4x(self):
        """<= 4x gain should use analog-only (multiplier=0, digital=0)."""
        reg = fx2driver._gain_db_to_register(6.0)  # 2x
        analog_mult = (reg >> 6) & 1
        digital_val = (reg >> 8) & 0x7F
        assert analog_mult == 0
        assert digital_val == 0


# ---------------------------------------------------------------------------
# Pure math: exposure rows
# ---------------------------------------------------------------------------


class TestExposureMath:
    """Exposure formula: SW = (tEXP + SO x 2 x tPIXCLK) / tROW, at the window's row."""

    def test_50ms_round_trip_within_one_row(self):
        column_size = fx2driver.IMG_WIDTH + 1
        rows = fx2driver.shutter_width_for(0.050, column_size)
        back = fx2driver.exposure_s(rows, column_size)
        assert abs(back - 0.050) < fx2driver.row_time_s(column_size), (
            f'50ms -> {rows} rows -> {back * 1000}ms (should be within 1 row of target)'
        )

    def test_max_rows_is_max_exposure(self):
        assert fx2driver.MAX_EXPOSURE_ROWS == 65535

    def test_the_shutter_width_lower_ceiling_is_the_data_sheets_at_the_full_window(self):
        max_ms = fx2driver.exposure_s(fx2driver.MAX_EXPOSURE_ROWS, fx2driver.IMG_WIDTH + 1) * 1000
        assert max_ms == pytest.approx(7934.6, abs=0.1)


# ---------------------------------------------------------------------------
# Frame delimiter parser
# ---------------------------------------------------------------------------


class TestFrameDelimiterParser:
    """The \\x01\\xfe\\x00\\xff delimiter pattern injected by GpifWaveform_Isr."""

    def test_delimiter_is_4_bytes(self):
        assert fx2driver.FRAME_DELIM == b'\x01\xfe\x00\xff'
        assert len(fx2driver.FRAME_DELIM) == 4

    def test_find_delimiter_at_known_offset(self):
        buf = bytearray(b'\xaa' * 100) + fx2driver.FRAME_DELIM + bytearray(b'\xbb' * 100)
        idx = buf.find(fx2driver.FRAME_DELIM)
        assert idx == 100

    def test_find_returns_negative_when_absent(self):
        buf = bytearray(b'\xaa' * 1000)
        assert buf.find(fx2driver.FRAME_DELIM) == -1

    def test_multiple_delimiters_finds_first(self):
        buf = (
            bytearray(b'\xaa' * 50)
            + fx2driver.FRAME_DELIM
            + bytearray(b'\xbb' * 50)
            + fx2driver.FRAME_DELIM
            + bytearray(b'\xcc' * 50)
        )
        assert buf.find(fx2driver.FRAME_DELIM) == 50


# ---------------------------------------------------------------------------
# Intel HEX parser
# ---------------------------------------------------------------------------


class TestIntelHexParser:
    """``parse_intel_hex`` is pure -- testable without USB or real firmware."""

    def test_parses_real_firmware(self):
        hex_path = REPO_ROOT / 'firmware' / 'LumascopeClassic.hex'
        if not hex_path.exists():
            pytest.skip(f'firmware hex not present at {hex_path}')
        data, end_addr = fx2driver.parse_intel_hex(str(hex_path))
        assert len(data) == 0x4000  # 16 KB 8051 program space
        assert end_addr > 0  # something was parsed
        assert end_addr <= 0x4000  # within program space
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
            'connect',
            'disconnect',
            'is_connected',
            'led_on',
            'led_off',
            'led_on_fast',
            'led_off_fast',
            'leds_off',
            'leds_off_fast',
            'leds_enable',
            'leds_disable',
            'available_channels',
            'read_led_current',
            'exchange_command',
        ]
        for name in required:
            assert hasattr(fx2driver.FX2LEDController, name), f'missing: {name}'

    def test_isinstance_check(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert isinstance(led, LEDBoardProtocol)


# ---------------------------------------------------------------------------
# FX2LEDController -- "thin command translator" regression guard
# ---------------------------------------------------------------------------


class TestFX2LEDThinTranslator:
    """The FX2 LED driver MUST NOT track LED state client-side.

    The LVC reference carried a ``led_ma`` dict that leaked the pre-4.1
    GUI's state-ownership model into the driver. In 4.1, the API owns
    state via ``Lumascope._led_owners``. These tests prove the new
    driver has no state bookkeeping:

    1. ``led_on`` writes the command and the driver answers no state query.
    2. The driver has no ``led_ma`` attribute (symbol-level regression guard).
    """

    def test_led_on_does_not_update_state_query(self, fake_fx2_conn):
        """The critical regression guard.

        Calling ``led_on(0, 100)`` must send I2C commands, and the driver
        must offer no state query to read back what it just did. If someone
        re-adds ``get_led_ma`` / ``is_led_on`` / ``get_led_state(s)``, a
        second store of LED state is back beside the API's.
        """
        led = fx2driver.FX2LEDController()
        led.led_on(0, 100)

        # The I2C command SHOULD have been sent (via fake conn.i2c_write)
        assert fake_fx2_conn.i2c_write.called, (
            'led_on should issue I2C writes through the FX2 connection'
        )

        for name in ('get_led_ma', 'is_led_on', 'get_led_state', 'get_led_states'):
            assert not hasattr(led, name), (
                f'the driver answers {name}; LED state belongs to the API'
            )

    def test_no_led_ma_attribute(self, fake_fx2_conn):
        """Symbol-level regression guard: the ``led_ma`` dict that the
        LVC reference driver carried must be absent in the new driver."""
        led = fx2driver.FX2LEDController()
        assert not hasattr(led, 'led_ma'), (
            'FX2LEDController.led_ma exists -- thin-translator rule violated'
        )


# ---------------------------------------------------------------------------
# available_channels (B3 integration)
# ---------------------------------------------------------------------------


class TestFX2LEDChannelDiscovery:
    """Post-B3 channel discovery -- `available_channels()` drives API validation
    and `ScopeCapabilities.led_channels`.
    """

    def test_returns_four_integer_channels(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        assert led.available_channels() == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Shared-connection invariant
# ---------------------------------------------------------------------------


class TestSharedConnectionInvariant:
    """FX2Camera and FX2LEDController must end up pointing at the SAME
    `_FX2Connection` singleton -- this is the whole point of the module-
    level singleton pattern, and is proven at the architecture level by
    TestRegistryAccommodatesCompositeHardware in test_driver_registry.py.
    Here we verify the real drivers satisfy it.
    """

    def test_camera_and_led_share_fx2_ref(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        cam = fx2driver.FX2Camera()
        assert led._fx2 is cam._fx2, 'camera and LED drivers must share the same _FX2Connection'

    def test_both_hold_same_fake(self, fake_fx2_conn):
        led = fx2driver.FX2LEDController()
        cam = fx2driver.FX2Camera()
        assert led._fx2 is fake_fx2_conn
        assert cam._fx2 is fake_fx2_conn


# ---------------------------------------------------------------------------
# FX2Camera profile loading + dynamic capabilities
# ---------------------------------------------------------------------------


class TestFX2FrameCallbackAdoption:
    """FX2 adopts the durable frame-callback contract: a callback registered
    before a reconnect survives the handler rebuild in connect()."""

    def test_reconnect_preserves_frame_callback(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()

        def _cb(frame):
            pass

        cam.register_frame_callback(_cb)
        assert _cb in cam.cam_image_handler._frame_callbacks

        # connect() rebuilds the handler (the reconnect path); the durable
        # registry must be re-applied onto the fresh _FX2ImageHandler.
        cam.connect()
        assert _cb in cam._registered_frame_callbacks
        assert _cb in cam.cam_image_handler._frame_callbacks


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
        assert cam.profile.gain.total_max_db == pytest.approx(42.1442, abs=1e-4)  # 128x
        assert cam.profile.exposure_min_us is not None
        assert cam.profile.exposure_max_us is not None

    def test_max_exposure_picks_up_dynamic_1000ms_cap(self, fake_fx2_conn):
        """Camera.max_exposure must reflect the dynamic 1000 ms cap, not
        the static MT9P031 register max (~7,366 ms). The UI exposure
        slider reads `ctx.max_exposure` (= camera.max_exposure) and was
        previously letting users dial past the safe-frame ceiling
        because Camera.max_exposure was a stored field set from a
        separate static profile field. Rule 2 fix: collapsed to a
        single source -- `profile.exposure_max_us` -- with
        `Camera.max_exposure` a derived property reading from it."""
        cam = fx2driver.FX2Camera()
        # exposure_max_us is overwritten to 1_000_000 by _query_dynamic_capabilities
        # (default from the profile entry is 7_366_000 = sensor register max).
        assert cam.profile.exposure_max_us == 1_000_000
        # max_exposure (in ms) is derived from exposure_max_us / 1000
        assert cam.max_exposure == 1000.0

    def test_pixel_format_is_mono8_only(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_pixel_format() == 'Mono8'
        assert cam.get_supported_pixel_formats() == ('Mono8',)

    def test_default_frame_size_is_1900x1900(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_frame_size() == {'width': 1900, 'height': 1900}

    def test_set_frame_size_returns_delivered_geometry(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        # 999 is off the grid of 4: the sensor acquires 1000 and the frame is
        # cropped back.
        delivered = cam.set_frame_size(1000, 999)
        assert delivered == {'width': 1000, 'height': 999}
        assert cam.get_frame_size() == {'width': 1000, 'height': 999}

    def test_set_frame_size_register_failure_returns_false_keeps_geometry(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        cam.set_frame_size(1000, 996)
        fake_fx2_conn.sensor_reg_write.side_effect = OSError('usb gone')
        try:
            result = cam.set_frame_size(500, 500)
        finally:
            fake_fx2_conn.sensor_reg_write.side_effect = None
        # A failed apply reports the base contract's False and leaves
        # get_frame_size() at the geometry the sensor is still running with.
        assert result is False
        assert cam.get_frame_size() == {'width': 1000, 'height': 996}

    def test_max_frame_size_is_1900x1900(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_max_frame_size() == {'width': 1900, 'height': 1900}

    def test_binning_returns_one(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        assert cam.get_binning_size() == 1

    @pytest.mark.parametrize(
        'call',
        [
            lambda cam: cam.auto_exposure_t(True),
            lambda cam: cam.auto_gain(True),
            lambda cam: cam.auto_gain_once(True),
            lambda cam: cam.update_auto_gain_target_brightness(0.5),
            lambda cam: cam.update_auto_gain_min_max(0.0, 30.0),
        ],
    )
    def test_auto_features_raise_rather_than_answer(self, fake_fx2_conn, call):
        """MT9P031 has no hardware AE/AG, and its profile says so; the API
        reads the profile and never asks. An answer here would claim a write
        that never happened, so a caller that skipped that read fails loudly.
        """
        cam = fx2driver.FX2Camera()
        with pytest.raises(NotImplementedError, match='no hardware auto-'):
            call(cam)

    def test_features_without_hardware_are_noops(self, fake_fx2_conn):
        cam = fx2driver.FX2Camera()
        cam.set_test_pattern(True, 'Black')
        cam.set_max_acquisition_frame_rate(True, 4.5)
        assert cam.get_all_temperatures() == {}


# ---------------------------------------------------------------------------
# camera_profiles.py registration
# ---------------------------------------------------------------------------


class TestCameraProfileRegistration:
    """The MT9P031 profile must be discoverable by the sensor name, the
    one key camera_profiles.py registers for it: FX2Camera sets the same
    model_name on every Classic model.
    """

    @pytest.mark.parametrize(
        'lookup_key',
        [
            'MT9P031',
            'MT9P031-LS620',  # what FX2Camera sets model_name to
        ],
    )
    def test_profile_found_by_substring(self, lookup_key):
        profile = lookup_profile(lookup_key)
        assert profile.driver == 'fx2', f'{lookup_key!r} should resolve to the FX2 profile'
        assert profile.sensor == 'Aptina MT9P031'
        assert profile.pixel_size_um == 2.2
        assert profile.pixel_formats == ['Mono8']
        assert profile.has_auto_gain is False
        assert profile.has_auto_exposure is False


# ---------------------------------------------------------------------------
# data/scopes.json shape for Classic models
# ---------------------------------------------------------------------------


class TestScopesJsonClassicModels:
    """LS620 and LS560 entries should exist with correct capability bits.

    The LS720 is the LS620's camera and LED with an XYZ stage on a
    TMCM-6110.
    """

    @pytest.fixture
    def scopes(self):
        path = REPO_ROOT / 'data' / 'scopes.json'
        with open(path) as f:
            return json.load(f)['Models']

    def test_ls620_exists(self, scopes):
        assert 'LS620' in scopes

    def test_ls560_exists(self, scopes):
        assert 'LS560' in scopes

    def test_ls720_has_an_xyz_stage_on_the_6110(self, scopes):
        entry = scopes['LS720']
        assert (entry['Focus'], entry['XYStage'], entry['Turret']) == (True, True, False)
        assert (entry['LEDBoard'], entry['MotorBoard']) == ('FX2', 'TMCM-6110')
        for key in ('Optics', 'Layers', 'Filterset'):
            assert entry[key] == scopes['LS620'][key]

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

    def test_ls620_has_fluorescence_bf(self, scopes):
        """LS620 carries the transmitted layer plus BGR fluorescence and
        NO separate PC layer -- PC on this model is BF with a mechanically
        installed phase slider, not a distinct illumination channel (the
        transmitted layer is displayed 'BF-Phase' for exactly that reason).
        Observed on hardware 2026-04-16: clicking a PhaseContrast layer
        raised `LED channel must be one of (0, 1, 2, 3), got -1` because
        FX2LEDController's colour table then had no 'PC' entry.
        """
        rows = {r['key_name']: r for r in scopes['LS620']['Layers']}
        assert set(rows) == {'BF', 'Blue', 'Green', 'Red'}
        assert rows['BF']['display_name'] == 'BF-Phase'
        assert rows['BF']['led_channel'] == 3
        assert 'PC' not in rows
        assert 'DF' not in rows

    def test_ls560_has_fluorescence_bf(self, scopes):
        """LS560 ships as the transmitted layer (displayed 'BF-Phase') plus
        Green only -- unlike LS620's BGR -- and no separate PC layer,
        same phase-slider rationale as test_ls620_has_fluorescence_bf.
        """
        rows = {r['key_name']: r for r in scopes['LS560']['Layers']}
        assert set(rows) == {'BF', 'Green'}
        assert rows['BF']['display_name'] == 'BF-Phase'
        assert rows['Green']['led_channel'] == 1
        assert 'PC' not in rows


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
        monkeypatch.setattr(fx2driver._FX2Connection, 'get', classmethod(lambda cls: fake))
        first = fx2driver._FX2Connection.get()
        second = fx2driver._FX2Connection.get()
        assert first is second

    def test_init_failure_leaves_instance_none(self):
        """If _FX2Connection.__init__ raises, the singleton stays None so
        the next attempt can retry from scratch (e.g., after plugging in
        the hardware).
        """

        def boom(self, transport):
            raise RuntimeError('no FX2 hardware')

        # Patch __init__ directly -- get() will still call cls() which
        # calls __new__ + __init__; if __init__ raises, cls._instance
        # is never assigned.
        with (
            patch.object(fx2driver._FX2Connection, '__init__', boom),
            pytest.raises(RuntimeError, match='no FX2 hardware'),
        ):
            fx2driver._FX2Connection.get()
        assert fx2driver._FX2Connection._instance is None


# ---------------------------------------------------------------------------
# WinUSB control-transfer failure contract
# ---------------------------------------------------------------------------
#
# `WinUsbDevice.control_transfer` used to discard the BOOL that
# WinUsb_ControlTransfer returns, so on the Windows streaming path a failed
# vendor request was indistinguishable from a successful one: the OUT branch
# returned None either way and the IN branch returned b'' because the
# transferred count stays 0 on failure. That None reached `_led_write`'s
# short-write detector, whose `result is not None` guard skipped the check --
# a safety check silently disabling itself.
#
# These drive the seam the consumers use rather than the ctypes wrapper:
# `drivers/winusb_iso.py` does `from ctypes import windll` at module scope and
# cannot be imported off-Windows at all. `control_transfer_out` reaches the
# WinUSB transport, which routes through its streaming reader's plain `device`
# attribute, so a stub reader substitutes for it on any platform.


def _conn_on_winusb(device):
    """A _FX2Connection on a WinUSB transport whose stream is running on a stub.

    Built with __new__ rather than the real constructor so the test needs no
    USB device; only the transport control_transfer_out routes through, plus
    the lock it takes.
    """
    transport = fx2driver._WinUsbTransport()
    transport._reader = SimpleNamespace(device=device)  # streaming: control goes here
    conn = object.__new__(fx2driver._FX2Connection)
    conn._lock = threading.Lock()
    conn._transport = transport
    return conn


def test_a_failed_winusb_out_transfer_reaches_the_caller_as_a_write_fault():
    """The contract every consumer in this chain is written against: a
    refused vendor write reaches the caller, typed, naming the request and
    carrying the transport's own words."""
    from drivers.exceptions import CameraWriteFailedError

    def refuse(*a, **kw):
        raise RuntimeError('ControlTransfer OUT ... failed: 31')

    conn = _conn_on_winusb(SimpleNamespace(control_transfer=refuse))
    with pytest.raises(CameraWriteFailedError, match=r'VR_I2C_WRITE.*failed: 31') as raised:
        conn.sensor_reg_write(0x42, 1)

    assert raised.value.reason == 'camera_write_failed'


@pytest.mark.parametrize('library', ['pyusb', 'libusb1', 'WinUSB'])
def test_a_vendor_write_the_fx2_does_not_answer_is_no_answer_on_every_path(library):
    """Every vendor write funnels through one place, and a timeout from any of
    the three libraries is the writer not answering: the camera's for a
    sensor write, the LED board's for an LED write."""
    from drivers.exceptions import BoardNoReplyError, CameraNoAnswerError

    error = _usb_timeouts()[library]

    def no_answer(*a, **kw):
        raise error

    conn = _conn_on_winusb(SimpleNamespace(control_transfer=no_answer))
    with pytest.raises(CameraNoAnswerError) as camera:
        conn.sensor_reg_write(0x42, 1)
    with pytest.raises(BoardNoReplyError) as led:
        conn.i2c_write(fx2driver.I2C_LED, [0xFF])

    assert (camera.value.reason, led.value.reason) == ('camera_no_answer', 'board_no_reply')
    assert str(error) in str(camera.value)
    assert str(error) in str(led.value)


def test_a_successful_winusb_out_transfer_returns_a_count_not_none():
    """`control_transfer_out` is declared `-> int` and its docstring promises
    bytes written. Returning None is what switched off the detector below."""
    conn = _conn_on_winusb(SimpleNamespace(control_transfer=lambda *a, **kw: 1))
    result = conn.i2c_write(fx2driver.I2C_LED, [0x01])
    assert result == 1
    assert result is not None


def test_the_led_short_write_detector_fires_on_a_zero_byte_write():
    """The test that would have failed for the whole life of the defect.

    `_led_write` compares i2c_write's return against the 1 byte it expects,
    but skips the comparison when the value is None -- which is all the
    WinUSB transport ever returned. With a real count flowing, a zero-byte
    write is caught.
    """
    led = object.__new__(fx2driver.FX2LEDController)
    led._fx2 = SimpleNamespace(i2c_write=lambda addr, data: 0)  # wrote nothing
    led._wire_debug_enabled = lambda: False

    with patch.object(fx2driver, 'logger') as mock_logger:
        led._led_write(channel=0, brightness=0x10)
        warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)

    assert 'short write' in warned, f'a 0-of-1-byte write was not reported: {warned!r}'


def test_the_led_detector_stays_quiet_on_a_good_write():
    """The converse -- the detector must not cry wolf on every LED command."""
    led = object.__new__(fx2driver.FX2LEDController)
    led._fx2 = SimpleNamespace(i2c_write=lambda addr, data: 1)
    led._wire_debug_enabled = lambda: False

    with patch.object(fx2driver, 'logger') as mock_logger:
        led._led_write(channel=0, brightness=0x10)
        warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)

    assert 'short write' not in warned


# ---------------------------------------------------------------------------
# A setting write the camera does not answer
# ---------------------------------------------------------------------------
#
# A USB timeout on the transfer that carries a setting is the camera not
# answering, not the camera refusing and not the camera gone: the API sends
# the write once more on that answer alone, so it has to arrive as its own
# type, CameraNoAnswerError, whichever library the transfer ran through, and
# it must not mark the camera removed. The FX2 reads nothing back, so every
# transfer a writer makes decides whether its write happened.


def _usb_timeouts():
    """One timeout from each library an FX2 control transfer runs through."""
    return {
        'pyusb': fx2driver.usb.core.USBTimeoutError('[Errno 60] Operation timed out'),
        'libusb1': fx2driver.usb1.USBErrorTimeout('LIBUSB_ERROR_TIMEOUT [-7]'),
        # WinUSB's ERROR_SEM_TIMEOUT, as drivers/winusb_iso.py raises it.
        'WinUSB': TimeoutError('ControlTransfer OUT (request=0xB3) timed out: 121'),
    }


@pytest.fixture
def simulated_fx2_camera():
    from drivers.simulated_fx2 import SimulatedFX2

    sim = SimulatedFX2()
    cam = fx2driver.FX2Camera(connection=sim.connection)
    yield sim, cam
    cam.disconnect()


def _every_transfer_times_out(monkeypatch, sim, error, reached):
    def no_answer(request, value, index, data, timeout):
        reached.append(request)
        raise error

    monkeypatch.setattr(sim.transport, 'control_out', no_answer)


def test_a_setting_write_the_fx2_does_not_answer_raises_no_answer(
    simulated_fx2_camera, monkeypatch
):
    from drivers.exceptions import CameraNoAnswerError

    sim, cam = simulated_fx2_camera
    # The twelve setting writers the API reaches, with a value each accepts.
    writers = {
        'gain': lambda: cam.gain(12.0),
        'exposure_t': lambda: cam.exposure_t(100.0),
        'auto_gain': lambda: cam.auto_gain(True),
        'auto_exposure_t': lambda: cam.auto_exposure_t(True),
        'set_frame_size': lambda: cam.set_frame_size(1000, 996),
        'set_binning_size': lambda: cam.set_binning_size(1),
        'set_pixel_format': lambda: cam.set_pixel_format('Mono8'),
        'set_conversion_gain_mode': lambda: cam.set_conversion_gain_mode('HCG'),
        'set_line_noise_reduction': lambda: cam.set_line_noise_reduction(True),
        'set_black_level': lambda: cam.set_black_level(0.0),
        'update_auto_gain_target_brightness': lambda: cam.update_auto_gain_target_brightness(0.5),
        'auto_gain_once': lambda: cam.auto_gain_once(True),
    }
    offenders = {}
    on_the_bus = set()
    for library, error in _usb_timeouts().items():
        for setting, write in writers.items():
            if not hasattr(cam, setting):
                continue  # not offered by the FX2; the API refuses it first
            reached = []
            with monkeypatch.context() as patched:
                _every_transfer_times_out(patched, sim, error, reached)
                try:
                    answer = write()
                    raised = None
                except Exception as e:
                    answer = raised = e
            name = f'{setting} ({library})'
            if not reached:
                if isinstance(raised, CameraNoAnswerError):
                    offenders[name] = 'no transfer was made, yet it answered no-answer'
                continue
            on_the_bus.add(setting)
            if not isinstance(raised, CameraNoAnswerError):
                offenders[name] = f'answered {answer!r}, not CameraNoAnswerError'
            elif raised.reason != 'camera_no_answer' or str(error) not in str(raised):
                offenders[name] = f'words or reason wrong: {raised.reason}: {raised}'
            if cam.is_device_removed() or sim.connection.removed:
                offenders[name] = 'the timeout marked the camera removed'
    assert not offenders, f'a setting write the FX2 did not answer: {offenders}'
    # The writers that reach the bus at all; the rest answer without it.
    assert on_the_bus == {'gain', 'exposure_t', 'set_frame_size'}


def test_a_gain_write_the_fx2_does_not_answer_leaves_the_gain_it_had(
    simulated_fx2_camera, monkeypatch
):
    from drivers.exceptions import CameraNoAnswerError

    sim, cam = simulated_fx2_camera
    before = cam.gain(6.0)
    _every_transfer_times_out(monkeypatch, sim, _usb_timeouts()['libusb1'], [])

    with pytest.raises(CameraNoAnswerError):
        cam.gain(30.0)

    assert cam.get_gain() == before


def test_a_frame_size_the_fx2_does_not_answer_stores_frames_as_they_arrive(
    simulated_fx2_camera, monkeypatch
):
    """Whether the window was applied is not known: the size reported is the
    one the camera last took, and no crop to the new one is promised."""
    from drivers.exceptions import CameraNoAnswerError

    sim, cam = simulated_fx2_camera
    cam.set_frame_size(1000, 996)
    _every_transfer_times_out(monkeypatch, sim, _usb_timeouts()['pyusb'], [])

    with pytest.raises(CameraNoAnswerError):
        cam.set_frame_size(500, 500)

    assert cam.get_frame_size() == {'width': 1000, 'height': 996}


class TestFx2DriverLibusbBackendProbe:
    """The FX2 gate resolves its libusb so a host without a usable one is
    classified as 'FX2 not applicable to this install' rather than crashing
    with NoBackendError mid-_connect. The library is libusb-package's bundled
    copy and never a host copy: each way the bundled load can fail --
    package absent, no library file, the backend bound to another file,
    python-libusb1 already holding another -- classifies unavailable, and
    the log says why.

    ``_fx2_gate`` is the decision the driver makes once at import; it is
    called here under a controlled fake usb / usb1 / libusb_package tree,
    since on this machine's real state it would be environment-dependent.
    """

    _BUNDLED = '/site-packages/libusb_package/libusb-1.0.dylib'

    @staticmethod
    def _backend_on(path):
        import types

        return types.SimpleNamespace(lib=types.SimpleNamespace(_name=path))

    @classmethod
    def _gate(
        cls,
        monkeypatch,
        *,
        backend,
        package=True,
        library_path=_BUNDLED,
        usb1_loads=True,
        usb1_opens=True,
        has_usb=True,
        has_usb1=True,
        platform='darwin',
    ):
        """The gate over a fake binding tree, with what the bindings were handed."""
        import ctypes
        import types

        # A stand-in by design: the subject is the gate's classification of
        # the bindings' answers, which no installed library can be made to give.
        usb_mod = types.ModuleType('usb')
        usb_backend = types.ModuleType('usb.backend')
        usb_libusb1 = types.ModuleType('usb.backend.libusb1')
        found = []

        def get_backend(find_library=None):
            found.append(find_library('usb-1.0') if find_library else None)
            return backend

        usb_libusb1.get_backend = get_backend
        usb_backend.libusb1 = usb_libusb1
        usb_mod.backend = usb_backend
        usb1_mod = types.ModuleType('usb1')
        handed_to_usb1 = []

        def load_library(lib):
            handed_to_usb1.append(lib)
            return usb1_loads

        usb1_mod.loadLibrary = load_library
        usb1_mod.getVersion = lambda: '1.0.30'

        # The fake library path cannot be opened; python-libusb1's own
        # handle is recorded by the path it was opened on instead.
        def open_handle(path):
            if not usb1_opens:
                raise OSError(f'dlopen({path}) failed')
            return ('CDLL', path)

        monkeypatch.setattr(ctypes, 'CDLL', open_handle)
        for name, mod in (
            ('usb', usb_mod),
            ('usb.backend', usb_backend),
            ('usb.backend.libusb1', usb_libusb1),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        if package:
            package_mod = types.ModuleType('libusb_package')
            package_mod.get_library_path = lambda: library_path
            monkeypatch.setitem(sys.modules, 'libusb_package', package_mod)
        else:
            # None in sys.modules makes the import raise ImportError.
            monkeypatch.setitem(sys.modules, 'libusb_package', None)

        gate = fx2driver._fx2_gate(has_usb, usb1_mod if has_usb1 else None, platform)
        return gate, found, handed_to_usb1

    @staticmethod
    def _logged(monkeypatch, gate):
        records = []

        class _Recorder:
            def __getattr__(self, level):
                return lambda msg, *a, **k: records.append((level.upper(), str(msg)))

        monkeypatch.setattr(fx2driver, 'logger', _Recorder())
        fx2driver._log_fx2_gate(gate)
        return records

    @staticmethod
    def _registered(monkeypatch, gate):
        """Whether the driver registers under ``gate``, and what a registry declining it was told."""
        monkeypatch.setattr(fx2driver, '_FX2_GATE', gate)
        registry = DriverRegistry('camera')

        class Dummy:
            pass

        fx2driver._register_if_fx2_available(registry, 'fx2', priority=80)(Dummy)
        return registry.registered_names()

    @pytest.mark.parametrize(
        ('case', 'kwargs', 'reason'),
        [
            ('package absent', {'package': False}, 'libusb-package is not installed'),
            ('no library file', {'library_path': None}, 'holds no library file'),
            ('backend did not load', {'backend': None}, 'did not load'),
            (
                'another libusb first',
                {'backend': 'OTHER'},
                'another libusb was loaded before the bundled one: /opt/homebrew/lib/libusb-1.0.dylib',
            ),
            (
                'usb1 holds another',
                {'usb1_loads': False},
                'python-libusb1 had already loaded another libusb',
            ),
            ('usb1 handle fails', {'usb1_opens': False}, 'did not load for python-libusb1'),
        ],
    )
    def test_each_failed_bundled_load_classifies_unavailable_and_says_why(
        self, monkeypatch, case, kwargs, reason
    ):
        backend = kwargs.pop('backend', self._backend_on(self._BUNDLED))
        if backend == 'OTHER':
            backend = self._backend_on('/opt/homebrew/lib/libusb-1.0.dylib')
        gate, _found, _handed = self._gate(monkeypatch, backend=backend, **kwargs)
        assert gate.library_path is None, case
        assert gate.available is False, case
        assert reason in gate.library_refusal, (case, gate.library_refusal)
        assert gate.unavailable_reason == f'bundled libusb not in use: {gate.library_refusal}'
        records = self._logged(monkeypatch, gate)
        assert any(
            lvl == 'INFO' and 'bundled libusb not in use' in msg and reason in msg
            for lvl, msg in records
        ), f'{case}: the refusal must be logged with its reason; got {records}'
        assert self._registered(monkeypatch, gate) == [], f'{case}: no driver may register'

    @pytest.mark.parametrize(
        ('case', 'kwargs', 'available', 'reason'),
        [
            ('no pyusb', {'has_usb': False}, False, 'pyusb is not installed'),
            ('no libusb1 off Windows', {'has_usb1': False}, False, 'libusb1 is not installed'),
            ('no libusb1 on Windows', {'has_usb1': False, 'platform': 'win32'}, True, None),
        ],
    )
    def test_the_bindings_each_platform_needs(self, monkeypatch, case, kwargs, available, reason):
        gate, _found, _handed = self._gate(
            monkeypatch, backend=self._backend_on(self._BUNDLED), **kwargs
        )
        assert (gate.available, gate.unavailable_reason) == (available, reason), case

    def test_the_bundled_library_binds_both_bindings_and_is_named(self, monkeypatch):
        """The backend is asked for the bundled file and nothing else, the
        gate opens, the drivers register, and the log names the file."""
        backend = self._backend_on(self._BUNDLED)
        gate, found, handed = self._gate(monkeypatch, backend=backend)
        assert handed == [('CDLL', self._BUNDLED)] and handed[0] is not backend.lib, (
            f'python-libusb1 must get its own handle to the bundled file; got {handed}'
        )
        assert found == [self._BUNDLED], (
            f'pyusb must be handed the bundled path, never a search; got {found}'
        )
        assert gate.library_path == self._BUNDLED
        assert gate.available is True
        assert ('INFO', f'[FX2 Driver] libusb 1.0.30 loaded from {self._BUNDLED}') in self._logged(
            monkeypatch, gate
        )
        assert self._registered(monkeypatch, gate) == ['fx2'], (
            'FX2 drivers must register when prerequisites are met'
        )
