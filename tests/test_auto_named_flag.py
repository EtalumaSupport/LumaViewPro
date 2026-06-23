# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Auto_Named flag behavior.

A step records explicitly whether it still carries an auto-generated name (vs a
user-typed one), instead of inferring that from the name string. This is the
structural fix behind the channel-change rename (#712/#719): an auto name
regenerates on a channel change, a user name is preserved, and the two are told
apart by the flag -- never by a lossy string match that could misread a user
name happening to match the auto pattern.
"""

from __future__ import annotations

import pathlib

from modules.protocol import Protocol
from tests.test_issue_524_extra_z_step_on_objective_change import (
    _add_layer_config,
    _empty_protocol_for_add,
)
from tests.test_protocol_roundtrip import _build_protocol, _make_step, _save_and_reload

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'

_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}
_WIDE_Z = {'Z': {'limits': {'min': 0.0, 'max': 10000.0}}}


def _insert_step(proto, *, step_name):
    proto.insert_step(
        step_name=step_name,
        layer='BF',
        layer_config=_add_layer_config(),
        plate_position={'x': 0.0, 'y': 0.0, 'z': 5000.0},
        objective_id='4x Oly',
        stim_configs={},
        before_step=0,
        after_step=None,
    )
    return proto.step(idx=0)


def test_auto_generated_step_is_flagged_auto():
    step = _insert_step(_empty_protocol_for_add(), step_name=None)
    assert step['Auto_Named']


def test_named_insert_is_not_auto():
    step = _insert_step(_empty_protocol_for_add(), step_name='my_step')
    assert not step['Auto_Named']


def test_modify_name_clears_auto_flag():
    proto = _build_protocol([_make_step(auto_named=True)])
    proto.modify_name(step_idx=0, step_name='renamed')
    assert not proto.step(idx=0)['Auto_Named']


def test_zstack_inherits_auto_flag():
    # A z-stack expansion copies the parent step; the auto flag must ride along
    # so the expanded slices keep the parent's auto-vs-user identity.
    proto = _build_protocol([_make_step(name='A1_BF', z=5000.0, z_slice=-1, auto_named=False)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=_WIDE_Z)
    flags = proto.steps()['Auto_Named'].tolist()
    assert flags and not any(flags), flags


def test_roundtrip_preserves_auto_flag(tmp_path):
    proto = _build_protocol(
        [
            _make_step(name='A1_BF', well='A1', auto_named=True),
            _make_step(name='B2_custom', well='B2', auto_named=False),
        ]
    )
    reloaded = _save_and_reload(proto, tmp_path / 'save')
    flags = list(reloaded.steps()['Auto_Named'].tolist())
    assert flags == [True, False], flags


def test_legacy_protocol_defaults_to_user_named(tmp_path):
    # A pre-v7 protocol carries no Auto_Named column; load defaults it to
    # user-named so a channel change never regenerates over a stored name.
    tsv = tmp_path / 'v6.tsv'
    tsv.write_text(
        'LumaViewPro Protocol\n'
        'Version\t6\n'
        'Period\t30.0\n'
        'Duration\t24.0\n'
        'Labware\t96 well Corning\n'
        '\n'
        'Steps\n'
        'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\t'
        'Auto_Gain\tExposure\tSum\tObjective\tWell\tTile\tZ-Slice\t'
        'Custom Step\tTile Group ID\tZ-Stack Group ID\tAcquire\t'
        'Video Config\tStim_Config\n'
        'A1_BF\t0\t0\t0\tFalse\tBF\tFalse\t50.0\t0.0\tFalse\t10.0\t1\t'
        "4x Oly\tA1\t\t-1\tFalse\t-1\t-1\timage\t{'fps': 5, 'duration': 5}\t{}\n"
    )
    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto is not None
    assert not proto.step(idx=0)['Auto_Named']
