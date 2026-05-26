# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the pure-Python helpers in tools/pylon_probe_sweep.py.

Hardware-touching paths (transport detection, cell apply, the actual
probe) require a connected Pylon camera; those are exercised by the
sweep tool itself on the bench. These tests cover the CLI argument
resolution + cell-product construction so a future refactor can't
silently drop a sweep dimension.
"""

import argparse

from tools.pylon_probe_sweep import (
    _build_gige_cells,
    _build_usb3_cells,
    _detect_transport,
    _format_cell_id,
    _resolve_resolutions,
)


class _FakeCamera:
    """Minimal stub for _detect_transport's model-name fallback path.

    `active` is truthy (a sentinel object) so the early-return on no-active
    doesn't fire; calling .GetTLNodeMap() raises, dropping the function
    into the model-name fallback that the regression test exercises.
    """

    class _ActiveStub:
        def GetTLNodeMap(self):
            raise RuntimeError('test stub: no TL node map')

    def __init__(self, model_name: str):
        self.active = self._ActiveStub()
        self.model_name = model_name


def _make_args(**overrides):
    """Minimal argparse.Namespace mirroring the tool's CLI defaults."""
    base = dict(
        pixel_formats=['Mono8'],
        resolution_tuples=[(2100, 2100)],
        dltl_modes=['On'],
        dltl_values_mb=[],
        gige_bw_modes=['Performance'],
        gige_packet_sizes=[1500],
        gige_delays=[0],
    )
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
        for cell, expected_mb in zip(cells, [160, 250, 360]):
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
