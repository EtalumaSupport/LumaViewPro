# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack or a tile grid is refused on a scope without the axis it moves.

Both are built by moving an axis, against that axis's travel limits. A
scope with no motor for the axis has no limits (its axes map is empty),
and the builders used to log an error and return an unchanged protocol --
so a caller that asked for a stack got a single plane and a status that
read as success. The build now refuses, once, in words the person reads,
with the reason a run that needs a missing axis already carries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.labware_loader import WellPlateLoader
from tests.test_run_refusal_contract import _capture_notifications
from tests.test_step_label_ssot import _ZSTACK, _WIDE_Z, _build_protocol, _labeled_step

UNREACHABLE = 'positions_unreachable'
_WIDE_XY = {
    'X': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
    'Y': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
}


def _one_step_protocol():
    return _build_protocol([_labeled_step()])


def _tile_2x2(proto, axes_config):
    return proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        curr_step_idx=0,
        axes_config=axes_config,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
        # Refused before the scale is read, so no real optics are needed.
        capabilities=SimpleNamespace(),
    )


@pytest.mark.parametrize('axes_config', [{}, _WIDE_XY], ids=['no motors', 'no Z'])
def test_a_zstack_without_z_is_refused_once_and_builds_nothing(monkeypatch, axes_config):
    proto = _one_step_protocol()
    before = proto.steps().copy()
    captured = _capture_notifications(monkeypatch)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=axes_config)

    assert refused.value.reason == UNREACHABLE
    assert 'Z' in refused.value.message
    assert len(captured) == 1
    assert proto.steps().equals(before)


@pytest.mark.parametrize('axes_config', [{}, _WIDE_Z], ids=['no motors', 'Z only'])
def test_a_tile_grid_without_xy_is_refused_once_and_builds_nothing(monkeypatch, axes_config):
    proto = _one_step_protocol()
    before = proto.steps().copy()
    captured = _capture_notifications(monkeypatch)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        _tile_2x2(proto, axes_config)

    assert refused.value.reason == UNREACHABLE
    assert 'X, Y' in refused.value.message
    assert len(captured) == 1
    assert proto.steps().equals(before)
