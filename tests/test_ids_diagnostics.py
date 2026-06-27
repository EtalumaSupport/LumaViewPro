"""Phase 4 (Pylon parity): IDS get_all_temperatures + read_diagnostic_snapshot.

The live counter values are bench-gated, but the data-shaping (temperature
assembly, snapshot dict shape, counter deltas + derived rates) is exercised
here against FindNode-style fakes -- the IDS Peak nodemap idiom the real
driver uses (FindNode().Value() for scalars, CurrentEntry()/AvailableEntries()
.SymbolicValue() for enums).
"""

import pytest

from tests.camera_fakes import bare_ids_camera


# --- FindNode-style fakes (mirror the ids_peak remote_nodemap API) ----------


class _Entry:
    def __init__(self, symbolic):
        self._symbolic = symbolic

    def SymbolicValue(self):
        return self._symbolic


class _ValueNode:
    def __init__(self, value):
        self._value = value

    def Value(self):
        return self._value


class _EnumNode:
    def __init__(self, symbolic):
        self._symbolic = symbolic

    def CurrentEntry(self):
        return _Entry(self._symbolic)


class _TempSelector:
    """DeviceTemperatureSelector: SetCurrentEntry steers DeviceTemperature."""

    def __init__(self, temps, current=None, entries=None):
        self._temps = temps
        # entries overrides what AvailableEntries reports (e.g. [] for a
        # vestigial selector); defaults to the temps keys.
        self._entries = list(temps) if entries is None else list(entries)
        self.current = current if current is not None else next(iter(temps))

    def AvailableEntries(self):
        return [_Entry(name) for name in self._entries]

    def CurrentEntry(self):
        return _Entry(self.current)

    def SetCurrentEntry(self, name):
        self.current = name


class _TempNode:
    def __init__(self, selector, temps):
        self._selector = selector
        self._temps = temps

    def Value(self):
        return self._temps[self._selector.current]


class _Nodemap:
    """FindNode dispatch. Names in ``missing`` raise (the ids_peak shape for an
    absent node); scalar names return _ValueNode, enum names _EnumNode."""

    def __init__(self, values=None, enums=None, special=None, missing=()):
        self._values = values or {}
        self._enums = enums or {}
        self._special = special or {}
        self._missing = set(missing)

    def FindNode(self, name):
        if name in self._missing:
            raise RuntimeError(f'simulated absent node: {name}')
        if name in self._special:
            return self._special[name]
        if name in self._enums:
            return _EnumNode(self._enums[name])
        if name in self._values:
            return _ValueNode(self._values[name])
        raise RuntimeError(f'simulated absent node: {name}')


def _temp_nodemap(temps=None, *, with_selector=True, missing=()):
    if temps is None:
        temps = {'Sensor': 42.5, 'FpgaCore': 55.0}
    special = {}
    if with_selector:
        selector = _TempSelector(temps)
        special['DeviceTemperatureSelector'] = selector
        special['DeviceTemperature'] = _TempNode(selector, temps)
    else:
        # Single-sensor body: only DeviceTemperature, no selector.
        special['DeviceTemperature'] = _ValueNode(next(iter(temps.values())))
        missing = (*missing, 'DeviceTemperatureSelector')
    return _Nodemap(special=special, missing=missing)


# --- get_all_temperatures ---------------------------------------------------


class TestGetAllTemperatures:
    def test_selector_iterates_every_entry(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _temp_nodemap({'Sensor': 42.5, 'FpgaCore': 55.0})
        assert cam.get_all_temperatures() == {'Sensor': 42.5, 'FpgaCore': 55.0}

    def test_single_sensor_body_reports_under_device(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _temp_nodemap({'only': 39.0}, with_selector=False)
        assert cam.get_all_temperatures() == {'Device': 39.0}

    def test_absent_temperature_node_returns_empty(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _Nodemap(missing=('DeviceTemperature',))
        assert cam.get_all_temperatures() == {}

    def test_inactive_camera_returns_empty(self):
        cam = bare_ids_camera()
        cam.active = False
        assert cam.get_all_temperatures() == {}


class TestGetSdkInfo:
    def test_ids_driver_names_its_sdk(self):
        cam = bare_ids_camera()
        info = cam.get_sdk_info()
        assert info['name'] == 'IDS peak'
        assert 'version' in info

    def test_sdk_less_driver_returns_unknown(self):
        # The base Camera default: a driver with no SDK reports name=None.
        from drivers.simulated_camera import SimulatedCamera

        assert SimulatedCamera().get_sdk_info() == {'name': None, 'version': None}

    def test_selector_is_restored_after_read(self):
        cam = bare_ids_camera()
        selector = _TempSelector({'Sensor': 42.5, 'FpgaCore': 55.0}, current='Sensor')
        cam.remote_nodemap = _Nodemap(
            special={
                'DeviceTemperatureSelector': selector,
                'DeviceTemperature': _TempNode(selector, {'Sensor': 42.5, 'FpgaCore': 55.0}),
            }
        )
        cam.get_all_temperatures()
        # Iteration left it at 'FpgaCore'; it must be restored to 'Sensor'.
        assert selector.current == 'Sensor'

    def test_empty_selector_falls_back_to_device(self):
        cam = bare_ids_camera()
        # Selector present but reports no entries; DeviceTemperature still reads.
        selector = _TempSelector({'Sensor': 42.5}, current='Sensor', entries=[])
        cam.remote_nodemap = _Nodemap(
            special={
                'DeviceTemperatureSelector': selector,
                'DeviceTemperature': _TempNode(selector, {'Sensor': 42.5}),
            }
        )
        assert cam.get_all_temperatures() == {'Device': 42.5}


# --- read_diagnostic_snapshot -----------------------------------------------


def _snapshot_nodemap():
    return _Nodemap(
        values={
            'DeviceModelName': 'U3-34L0XCP-M',
            'DeviceSerialNumber': '4108888',
            'DeviceFirmwareVersion': '3.80.26143',
            'DeviceVersion': '1.2',
            'Width': 1900,
            'Height': 1900,
            'ExposureTime': 5000.0,
            'Gain': 1.0,
            'DeviceLinkThroughputLimit': 420000000,
            'AcquisitionFrameRate': 31.4,
            'PayloadSize': 7220000,
            'BinningVertical': 1,
            'BinningHorizontal': 1,
        },
        enums={'PixelFormat': 'Mono10g40IDS', 'DeviceLinkThroughputLimitComponent': 'Link'},
    )


class TestReadDiagnosticSnapshot:
    def test_inactive_camera_reports_disconnected_but_supported(self):
        cam = bare_ids_camera()
        cam.active = False
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert snap['supported'] is True
        assert snap['connected'] is False
        assert 'camera not connected' in snap['errors']

    def test_identity_and_config_populated_from_nodemap(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _snapshot_nodemap()
        cam.remote_nodemap._special = {'DeviceTemperatureSelector': None}  # no temps
        cam.get_all_temperatures = lambda: {}
        cam._read_stream_stats = lambda: {}
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert snap['connected'] is True
        assert snap['camera']['model_name'] == 'U3-34L0XCP-M'
        assert snap['camera']['serial'] == '4108888'
        assert snap['config']['width'] == 1900
        assert snap['config']['pixel_format'] == 'Mono10g40IDS'
        assert snap['config']['dltl_component'] == 'Link'

    def test_missing_config_node_records_sentinel_not_raises(self):
        cam = bare_ids_camera()
        # A nodemap missing every config node: each read records a sentinel.
        cam.remote_nodemap = _Nodemap()
        cam.get_all_temperatures = lambda: {}
        cam._read_stream_stats = lambda: {}
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert snap['connected'] is True
        assert str(snap['camera']['model_name']).startswith('<missing')
        assert str(snap['config']['width']).startswith('<missing')

    def test_counter_deltas_computed_but_no_derived_without_window(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _snapshot_nodemap()
        cam.get_all_temperatures = lambda: {}
        polls = [
            {'StreamDeliveredFrameCount': 100, 'StreamLostFrameCount': 5},
            {'StreamDeliveredFrameCount': 160, 'StreamLostFrameCount': 8},
        ]
        cam._read_stream_stats = lambda: polls.pop(0)
        # duration_s=0: deltas are window-independent, but no derived rates are
        # emitted (no real sampling window elapsed).
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert snap['deltas']['StreamDeliveredFrameCount'] == 60
        assert snap['deltas']['StreamLostFrameCount'] == 3
        assert snap['derived'] == {}

    def test_derived_rates_over_real_window(self, monkeypatch):
        from types import SimpleNamespace

        import drivers.idscamera as idscamera

        cam = bare_ids_camera()
        cam.remote_nodemap = _snapshot_nodemap()
        cam.get_all_temperatures = lambda: {}
        polls = [
            {'StreamDeliveredFrameCount': 100, 'StreamLostFrameCount': 5},
            {'StreamDeliveredFrameCount': 160, 'StreamLostFrameCount': 8},
        ]
        cam._read_stream_stats = lambda: polls.pop(0)
        # Deterministic 3.0s window: mock monotonic + no-op sleep.
        clock = iter([1000.0, 1003.0])
        monkeypatch.setattr(
            idscamera, 'time', SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda s: None)
        )
        snap = cam.read_diagnostic_snapshot(duration_s=3.0)
        assert snap['derived']['observed_fps'] == pytest.approx(60 / 3.0)
        assert snap['derived']['losses_per_second'] == pytest.approx(3 / 3.0)
        assert snap['derived']['loss_rate_pct'] == pytest.approx(100.0 * 3 / 63)

    def test_counter_reset_midwindow_yields_no_negative_delta(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _snapshot_nodemap()
        cam.get_all_temperatures = lambda: {}
        # post < pre: a StartAcquisition reset the counters mid-window.
        polls = [
            {'StreamDeliveredFrameCount': 500, 'StreamLostFrameCount': 9},
            {'StreamDeliveredFrameCount': 40, 'StreamLostFrameCount': 0},
        ]
        cam._read_stream_stats = lambda: polls.pop(0)
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert snap['deltas']['StreamDeliveredFrameCount'] is None
        assert snap['deltas']['StreamLostFrameCount'] is None

    def test_stats_access_error_is_self_reported_and_deltas_none(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _snapshot_nodemap()
        cam.get_all_temperatures = lambda: {}
        # Real _read_stream_stats path: NodeMaps() raises -> self-reported.
        cam.data_stream.NodeMaps.side_effect = RuntimeError('no stream nodemap')
        snap = cam.read_diagnostic_snapshot(duration_s=0)
        assert '_access_error' in snap['stats_pre']
        assert all(v is None for v in snap['deltas'].values())
