# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the pure-Python helpers in tools/pylon_probe_sweep.py.

The actual probe requires a connected Pylon camera and is exercised by the
sweep tool itself on the bench. These tests cover the CLI argument
resolution + cell-product construction so a future refactor can't
silently drop a sweep dimension, the transport detection's answers, a
cell's writes through the simulated scope's own setters, and a cell's
record of a setting the camera refuses.
"""

import argparse
from types import SimpleNamespace

import pytest

from _lumascope.drivers.camera import CameraNodeValue
from _lumascope.drivers.exceptions import CameraNoAnswerError, CameraWriteFailedError
from tools.pylon_probe_sweep import (
    _apply_cell,
    _build_gige_cells,
    _build_usb3_cells,
    _detect_transport,
    _format_cell_id,
    _resolve_resolutions,
    _sweep,
)


class _NodeScope:
    """A scope whose camera answers ``nodes`` by name, None for any other.

    a stand-in by design: no simulated camera is a Pylon camera (the
    simulated one has no named settings), so the GigE and USB3 answers
    are a simulator gap; the subject is the tool's reading of them.
    """

    def __init__(self, nodes):
        self.capabilities = SimpleNamespace(camera_node_maps=('device', 'stream'))
        self.diagnostics = SimpleNamespace(read_camera_node=lambda name: nodes.get(name))


def _make_args(**overrides):
    """Minimal argparse.Namespace mirroring the tool's CLI defaults."""
    base = {
        'image_modes': ['8bit'],
        'resolution_tuples': [(2100, 2100)],
        'dltl_modes': ['On'],
        'dltl_values_mb': [],
        'gige_bw_modes': ['Performance'],
        'gige_packet_sizes': [1500],
        'gige_delays': [0],
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestResolveResolutions:
    def test_integer_token_expands_to_square(self):
        assert _resolve_resolutions(['2100'], (4000, 3000)) == [(2100, 2100)]

    def test_wxh_token(self):
        assert _resolve_resolutions(['1920x1080'], (4000, 3000)) == [(1920, 1080)]

    def test_sensor_max_token(self):
        assert _resolve_resolutions(['sensor-max'], (4096, 3000)) == [(4096, 3000)]

    def test_mixed_tokens_keep_order(self):
        result = _resolve_resolutions(['1900', '2100', 'sensor-max'], (4096, 3000))
        assert result == [(1900, 1900), (2100, 2100), (4096, 3000)]

    def test_non_numeric_token_exits_cleanly_not_traceback(self, capsys):
        """Bench 2026-05-26: operator dropped '--' before --dltl-modes so
        argparse's nargs='+' for --resolutions consumed 'dltl-modes' as a
        token. Pre-fix: bare int(tok) raised ValueError -> CRITICAL crash
        traceback in lumaviewpro_errors.log. Post-fix: clean stderr message
        + sys.exit(2) so the operator sees what went wrong without parsing
        a Python traceback."""
        import pytest as _pytest

        with _pytest.raises(SystemExit) as excinfo:
            _resolve_resolutions(['2100', 'dltl-modes'], (4000, 3000))
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert 'dltl-modes' in captured.err
        assert '--' in captured.err  # mentions the missing-dash diagnosis


class TestUsb3CellMatrix:
    def test_default_one_cell(self):
        cells = _build_usb3_cells(_make_args())
        assert cells == [
            {
                'image_mode': '8bit',
                'resolution': (2100, 2100),
                'dltl_mode': 'On',
                'dltl_value': None,
            }
        ]

    def test_dltl_off_emits_one_cell_per_pf_res(self):
        cells = _build_usb3_cells(_make_args(dltl_modes=['Off']))
        assert len(cells) == 1
        assert cells[0]['dltl_mode'] == 'Off'
        assert cells[0]['dltl_value'] is None

    def test_dltl_on_with_values_expands(self):
        cells = _build_usb3_cells(_make_args(dltl_modes=['On'], dltl_values_mb=[160, 250, 360]))
        assert len(cells) == 3
        for cell, expected_mb in zip(cells, [160, 250, 360], strict=False):
            assert cell['dltl_mode'] == 'On'
            assert cell['dltl_value'] == expected_mb * 1_000_000

    def test_full_cartesian(self):
        cells = _build_usb3_cells(
            _make_args(
                image_modes=['8bit', '12bit_scientific'],
                resolution_tuples=[(1900, 1900), (2100, 2100)],
                dltl_modes=['Off', 'On'],
                dltl_values_mb=[160, 360],
            )
        )
        # 2 modes x 2 res x (1 Off + 2 On values) = 12
        assert len(cells) == 12


class TestGigeCellMatrix:
    def test_default_one_cell(self):
        cells = _build_gige_cells(_make_args())
        assert cells == [
            {
                'image_mode': '8bit',
                'resolution': (2100, 2100),
                'bw_mode': 'Performance',
                'packet_size': 1500,
                'delay_ticks': 0,
            }
        ]

    def test_full_cartesian(self):
        cells = _build_gige_cells(
            _make_args(
                image_modes=['8bit'],
                resolution_tuples=[(2100, 2100)],
                gige_bw_modes=['Default', 'Performance'],
                gige_packet_sizes=[1500, 9000],
                gige_delays=[0, 100],
            )
        )
        # 1 mode x 1 res x 2 bw x 2 pkt x 2 delay = 8
        assert len(cells) == 8

    def test_packet_sizes_coerced_to_int(self):
        cells = _build_gige_cells(_make_args(gige_packet_sizes=['9000']))
        assert cells[0]['packet_size'] == 9000


class TestFormatCellId:
    def test_usb3_dltl_off(self):
        cell = {
            'image_mode': '8bit',
            'resolution': (2100, 2100),
            'dltl_mode': 'Off',
            'dltl_value': None,
        }
        out = _format_cell_id('usb3', cell)
        assert '8bit' in out
        assert '2100x2100' in out
        assert 'dltl=Off' in out

    def test_usb3_dltl_on_with_value(self):
        cell = {
            'image_mode': '8bit',
            'resolution': (1900, 1900),
            'dltl_mode': 'On',
            'dltl_value': 160_000_000,
        }
        out = _format_cell_id('usb3', cell)
        assert 'dltl=160M' in out

    def test_gige(self):
        cell = {
            'image_mode': '8bit',
            'resolution': (2100, 2100),
            'bw_mode': 'Performance',
            'packet_size': 9000,
            'delay_ticks': 50,
        }
        out = _format_cell_id('gige', cell)
        assert 'bw=Performance' in out
        assert 'pkt=9000' in out
        assert 'spcd=50' in out


def test_sensor_max_on_a_scope_that_knows_no_largest_frame_exits_cleanly(capsys):
    """No frame stands in for the largest one the scope does not know."""
    with pytest.raises(SystemExit) as excinfo:
        _resolve_resolutions(['sensor-max'], None)
    assert excinfo.value.code == 2
    assert 'frame_max is None' in capsys.readouterr().err


class TestDetectTransport:
    """The camera's own nodes say its transport."""

    def test_a_body_with_a_gev_packet_size_is_gige(self):
        # A GigE body has a throughput limit too: the GEV node decides.
        scope = _NodeScope(
            {
                'GevSCPSPacketSize': CameraNodeValue(1500, 'B'),
                'DeviceLinkThroughputLimitMode': CameraNodeValue('On', ''),
            }
        )
        assert _detect_transport(scope) == 'gige'

    def test_a_body_with_a_throughput_limit_and_no_gev_node_is_usb3(self):
        scope = _NodeScope({'DeviceLinkThroughputLimitMode': CameraNodeValue('On', '')})
        assert _detect_transport(scope) == 'usb3'

    def test_a_body_with_neither_node_is_unknown(self):
        assert _detect_transport(_NodeScope({})) == 'unknown'

    def test_a_camera_with_no_named_settings_is_unknown(self, sim_scope):
        assert _detect_transport(sim_scope) == 'unknown'


def test_a_cell_sets_the_image_mode_and_frame_as_lvp_does(sim_scope):
    """The image mode reaches the camera as the pixel format LVP captures it
    in, and the frame as LVP frames it; a knob this camera does not offer is
    the cell's record (FAILED, not offered), not an abort."""
    largest = sim_scope.capabilities.camera_limits.frame_max
    cell = {
        'image_mode': '12bit_scientific',
        'resolution': (largest.width, largest.height),
        'dltl_mode': 'On',
        'dltl_value': 160_000_000,
    }

    log = _apply_cell(sim_scope, 'usb3', cell)

    assert log == [
        ('image_mode', 'Mono12'),
        ('frame', f'{largest.width}x{largest.height}'),
        (
            'dltl',
            'FAILED: set_device_link_throughput_limit was not written: '
            'SimulatedCamera does not offer it',
        ),
    ]
    assert sim_scope.camera.read_frame_size() == largest


def test_a_frame_past_the_largest_is_set_at_it_and_the_cell_records_what_was_delivered(
    sim_scope,
):
    """The scope sets a side past its largest frame at it (and tells it), so
    the cell's record is the frame delivered, not a refusal."""
    largest = sim_scope.capabilities.camera_limits.frame_max
    cell = {'image_mode': '8bit', 'resolution': (largest.width + 2, largest.height)}

    log = _apply_cell(sim_scope, 'unknown', cell)

    assert log[:2] == [('image_mode', 'Mono8'), ('frame', f'{largest.width}x{largest.height}')]
    assert sim_scope.camera.read_frame_size() == largest


def test_a_sweep_on_a_camera_whose_transport_is_unknown_stops_before_any_cell(sim_scope, capsys):
    args = argparse.Namespace(resolutions=['sensor-max'], host_load_workers=0, **vars(_make_args()))
    with pytest.raises(SystemExit) as excinfo:
        _sweep(sim_scope, args)
    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "unknown transport 'unknown'" in out
    assert '[1/' not in out


def test_a_knob_the_camera_refuses_is_the_cells_record_and_the_cell_goes_on():
    """A GEV write the camera refuses, or does not answer, is recorded as that
    knob's FAILED entry, as a refused image mode is, and the knobs after it
    are set."""
    calls = []

    def refused(name):
        def set_knob(value):
            calls.append(name)
            raise CameraWriteFailedError(f'{name} not writable now')

        return set_knob

    def unanswered(name):
        def set_knob(value):
            calls.append(name)
            raise CameraNoAnswerError(f'{name} not answered')

        return set_knob

    def taken(name):
        def set_knob(value):
            calls.append(name)

        return set_knob

    # The API's bench-knob setters answer for a Pylon camera only.
    # a stand-in by design: the subject is the tool's catch around those setters.
    scope = SimpleNamespace(
        camera=SimpleNamespace(
            set_image_mode=lambda mode: None,
            read_pixel_format=lambda: 'Mono8',
            set_frame_size=lambda w, h: SimpleNamespace(width=w, height=h),
        ),
        imaging=SimpleNamespace(
            _set_bandwidth_reserve_mode=taken('bw_mode'),
            _set_gev_packet_size=refused('packet_size'),
            _set_gev_inter_packet_delay=unanswered('delay_ticks'),
        ),
    )
    cell = {
        'image_mode': '8bit',
        'resolution': (2100, 2100),
        'bw_mode': 'Performance',
        'packet_size': 9000,
        'delay_ticks': 0,
    }

    log = _apply_cell(scope, 'gige', cell)

    assert calls == ['bw_mode', 'packet_size', 'delay_ticks']
    assert log == [
        ('image_mode', 'Mono8'),
        ('frame', '2100x2100'),
        ('bw_mode', 'written'),
        ('packet_size', 'FAILED: packet_size not writable now'),
        ('delay_ticks', 'FAILED: delay_ticks not answered'),
    ]
