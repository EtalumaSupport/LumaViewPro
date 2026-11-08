# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the pure-Python helpers in tools/pylon_probe_sweep.py.

The actual probe requires a connected Pylon camera and is exercised by the
sweep tool itself on the bench. These tests cover the CLI argument
resolution + cell-product construction so a future refactor can't
silently drop a sweep dimension, the transport detection's answers, and
a cell's record of a knob the camera refuses.
"""

import argparse
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from _lumascope.drivers.camera import CameraNodeHasNoValue, CameraNodeValue
from _lumascope.drivers.exceptions import CameraReadFailedError, CameraWriteFailedError
from tools.pylon_probe_sweep import (
    _apply_cell,
    _build_gige_cells,
    _build_usb3_cells,
    _detect_transport,
    _format_cell_id,
    _resolve_resolutions,
)


class _FakeCamera:
    """A connected camera whose transport node answers ``transport_node``.

    a stand-in by design: the subject is the tool's reading of the driver's
    ``read_node`` answer, and no simulated camera is a Pylon camera. By
    default the camera does not offer DeviceTransportLayerType (``None``),
    the model-name fallback's case; an exception is raised by the read.
    """

    def __init__(self, model_name: str, transport_node=None):
        self.active = object()
        self.model_name = model_name
        self._transport_node = transport_node

    def read_node(self, name, node_map):
        assert (name, node_map) == ('DeviceTransportLayerType', 'transport')
        if isinstance(self._transport_node, BaseException):
            raise self._transport_node
        return self._transport_node


def _make_args(**overrides):
    """Minimal argparse.Namespace mirroring the tool's CLI defaults."""
    base = {
        'pixel_formats': ['Mono8'],
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
        assert _resolve_resolutions(['2100'], 4000, 3000) == [(2100, 2100)]

    def test_wxh_token(self):
        assert _resolve_resolutions(['1920x1080'], 4000, 3000) == [(1920, 1080)]

    def test_sensor_max_token(self):
        assert _resolve_resolutions(['sensor-max'], 4096, 3000) == [(4096, 3000)]

    def test_mixed_tokens_keep_order(self):
        result = _resolve_resolutions(['1900', '2100', 'sensor-max'], 4096, 3000)
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
            _resolve_resolutions(['2100', 'dltl-modes'], 4000, 3000)
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert 'dltl-modes' in captured.err
        assert '--' in captured.err  # mentions the missing-dash diagnosis


class TestUsb3CellMatrix:
    def test_default_one_cell(self):
        cells = _build_usb3_cells(_make_args())
        assert cells == [
            {
                'pixel_format': 'Mono8',
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
                pixel_formats=['Mono8', 'Mono12'],
                resolution_tuples=[(1900, 1900), (2100, 2100)],
                dltl_modes=['Off', 'On'],
                dltl_values_mb=[160, 360],
            )
        )
        # 2 pf x 2 res x (1 Off + 2 On values) = 12
        assert len(cells) == 12


class TestGigeCellMatrix:
    def test_default_one_cell(self):
        cells = _build_gige_cells(_make_args())
        assert cells == [
            {
                'pixel_format': 'Mono8',
                'resolution': (2100, 2100),
                'bw_mode': 'Performance',
                'packet_size': 1500,
                'delay_ticks': 0,
            }
        ]

    def test_full_cartesian(self):
        cells = _build_gige_cells(
            _make_args(
                pixel_formats=['Mono8'],
                resolution_tuples=[(2100, 2100)],
                gige_bw_modes=['Default', 'Performance'],
                gige_packet_sizes=[1500, 9000],
                gige_delays=[0, 100],
            )
        )
        # 1 pf x 1 res x 2 bw x 2 pkt x 2 delay = 8
        assert len(cells) == 8

    def test_packet_sizes_coerced_to_int(self):
        cells = _build_gige_cells(_make_args(gige_packet_sizes=['9000']))
        assert cells[0]['packet_size'] == 9000


class TestFormatCellId:
    def test_usb3_dltl_off(self):
        cell = {
            'pixel_format': 'Mono8',
            'resolution': (2100, 2100),
            'dltl_mode': 'Off',
            'dltl_value': None,
        }
        out = _format_cell_id('usb3', cell)
        assert 'Mono8' in out
        assert '2100x2100' in out
        assert 'dltl=Off' in out

    def test_usb3_dltl_on_with_value(self):
        cell = {
            'pixel_format': 'Mono8',
            'resolution': (1900, 1900),
            'dltl_mode': 'On',
            'dltl_value': 160_000_000,
        }
        out = _format_cell_id('usb3', cell)
        assert 'dltl=160M' in out

    def test_gige(self):
        cell = {
            'pixel_format': 'Mono8',
            'resolution': (2100, 2100),
            'bw_mode': 'Performance',
            'packet_size': 9000,
            'delay_ticks': 50,
        }
        out = _format_cell_id('gige', cell)
        assert 'bw=Performance' in out
        assert 'pkt=9000' in out
        assert 'spcd=50' in out


class TestDetectTransportModelFallback:
    """Model-name fallback must classify Basler ACE + ACE 2 + GigE families."""

    def test_basler_ace_usb_mono(self):
        assert _detect_transport(_FakeCamera('daA3840-45um')) == 'usb3'

    def test_basler_ace_usb_color(self):
        assert _detect_transport(_FakeCamera('daA3840-45uc')) == 'usb3'

    def test_basler_ace_gige_mono(self):
        assert _detect_transport(_FakeCamera('dmA3536-9gm')) == 'gige'

    def test_basler_ace_gige_color(self):
        assert _detect_transport(_FakeCamera('dmA3536-9gc')) == 'gige'

    def test_basler_ace2_usb_mono_with_bas_series(self):
        # Lumi-board camera; earlier 4-char tail check missed this and
        # produced "ERROR: unknown transport 'unknown'" at bench launch.
        assert _detect_transport(_FakeCamera('a2A3536-31umBAS')) == 'usb3'

    def test_basler_ace2_usb_color_with_bas_series(self):
        assert _detect_transport(_FakeCamera('a2A3840-45ucBAS')) == 'usb3'

    def test_unknown_model_returns_unknown(self):
        assert _detect_transport(_FakeCamera('foobar-vendor')) == 'unknown'

    def test_empty_model_returns_unknown(self):
        assert _detect_transport(_FakeCamera('')) == 'unknown'


def test_the_transport_node_is_read_before_the_model_name():
    camera = _FakeCamera('daA3840-45um', CameraNodeValue('GigEVision', ''))
    assert _detect_transport(camera) == 'gige'


def test_a_transport_the_camera_cannot_state_ends_the_sweep():
    """A failed read, or a transport node that holds no name, raises: the
    sweep cannot pick its cells for a transport it does not know, and the
    model name is the answer only for a camera that lacks the node."""
    failed = _FakeCamera('daA3840-45um', CameraReadFailedError('usb gone'))
    with pytest.raises(CameraReadFailedError, match='usb gone'):
        _detect_transport(failed)

    no_value = _FakeCamera('daA3840-45um', CameraNodeHasNoValue.NOT_READABLE_NOW)
    with pytest.raises(CameraReadFailedError, match='holds no transport name'):
        _detect_transport(no_value)


def test_a_knob_the_camera_refuses_is_the_cells_record_and_the_cell_goes_on():
    """A GEV write the camera refuses is recorded as that knob's FAILED
    entry, as a refused pixel format is, and the knobs after it are set."""
    calls = []

    def refused(name):
        def set_knob(value):
            calls.append(name)
            raise CameraWriteFailedError(f'{name} not writable now')

        return set_knob

    def taken(name):
        def set_knob(value):
            calls.append(name)
            return True

        return set_knob

    # a stand-in by design: the subject is the tool's catch around the
    # API's bench-knob setters, which a Pylon camera answers.
    scope = SimpleNamespace(
        camera=SimpleNamespace(update_camera_config=nullcontext),
        imaging=SimpleNamespace(
            set_pixel_format=lambda pf: None,
            _set_bandwidth_reserve_mode=taken('bw_mode'),
            _set_gev_packet_size=refused('packet_size'),
            _set_gev_inter_packet_delay=taken('delay_ticks'),
        ),
    )
    cell = {
        'pixel_format': 'Mono8',
        'resolution': (0, 0),
        'bw_mode': 'Performance',
        'packet_size': 9000,
        'delay_ticks': 0,
    }

    log = _apply_cell(scope, 'gige', cell, 0, 0)

    assert calls == ['bw_mode', 'packet_size', 'delay_ticks']
    assert log == [
        ('pixel_format', 'Mono8'),
        ('bw_mode', True),
        ('packet_size', 'FAILED: packet_size not writable now'),
        ('delay_ticks', True),
    ]
