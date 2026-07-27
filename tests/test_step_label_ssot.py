# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Label as a persisted column -- the single source of truth for a step's base text.

A protocol step's user-visible base text travels as the persisted Label
column; Name is a DERIVED display column re-rendered from the structured
columns (Label, Well, Color, Tile, Z-Slice) at every mutation and at load.
The old design recovered the base by re-parsing the rendered Name through
the token vocabulary, which truncated any user text with a token-shaped
segment ('Treatment_10x' -> 'Treatment') and collapsed distinct steps onto
one output filename.

Covers:
- writer-derived filename bases keep full user labels distinct (T1)
- modify_name on a well step derives Name from the label, not the well (T2)
- v7 -> v8 load migration recovers labels and re-stamps the version (T3)
- pre-v7 compose-and-compare recovers the auto/user flag per row (T4)
- tiling / z-stack children inherit the parent label (T5)
- labels sit outside the token vocabulary's blast radius (T7)
- post-record v1 -> v2 in-place file upgrade with Label back-fill
"""

from __future__ import annotations

import datetime
import pathlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from modules.common_utils import build_step_name, recover_step_label, step_components
from modules.exceptions import ProtocolError
from modules.protocol import Protocol, ProtocolFormatError
from modules.protocol_post_record import ProtocolPostRecord

from tests.test_protocol_roundtrip import _build_protocol, _make_step

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


def _writer_base(step, scan_count=0) -> str:
    """The filename base exactly as the capture writer derives it."""
    return build_step_name(step_components(step, scan_count=scan_count))


def _step_row(label, color='BF', well=''):
    return {'Well': well, 'Label': label, 'Color': color, 'Tile': '', 'Z-Slice': -1}


# ---------------------------------------------------------------------------
# T1: user labels with token-shaped segments stay whole and distinct in the
# writer-derived filename base.
# ---------------------------------------------------------------------------


def test_writer_bases_keep_full_labels_distinct():
    labels = ['Treatment_10x', 'Treatment_20x', 'Sample_2026', 'Culture_video']
    bases = [_writer_base(_step_row(label)) for label in labels]
    assert len(set(bases)) == len(labels), f'bases must be distinct: {bases}'
    for label, base in zip(labels, bases, strict=True):
        assert label in base, f'base {base!r} must contain the full label {label!r}'
    # The exact pre-fix collapse: 'Treatment_10x' and 'Treatment_20x' both
    # parsed down to 'Treatment' and landed on one filename.
    assert bases[0] != bases[1]
    assert bases[0] == 'Treatment_10x_BF_0000'
    assert bases[1] == 'Treatment_20x_BF_0000'


# ---------------------------------------------------------------------------
# T2: renaming a well step derives its Name (and filename base) from the
# label, not the well anchor.
# ---------------------------------------------------------------------------


def test_modify_name_on_well_step_renders_label_base():
    proto = _build_protocol(
        [
            _make_step(
                name='A2_BF',
                well='A2',
                z_slice=-1,
                tile_group_id=-1,
                zstack_group_id=-1,
            )
        ]
    )
    proto.modify_name(step_idx=0, step_name='Control_preDose')
    step = proto.step(idx=0)
    assert step['Label'] == 'Control_preDose'
    assert step['Name'] == 'Control_preDose_BF'
    assert not step['Auto_Named']
    base = _writer_base(step)
    assert base.startswith('Control_preDose_BF'), base
    assert not base.startswith('A2_'), base


# ---------------------------------------------------------------------------
# T3 / T4: load-boundary migration.
# ---------------------------------------------------------------------------

_V7_TSV = (
    'LumaViewPro Protocol\n'
    'Version\t7\n'
    'Period\t0\n'
    'Duration\t0\n'
    'Labware\tCenter Plate\n'
    'Capture Root\t\n'
    'Steps\n'
    'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\tAuto_Gain\t'
    'Exposure\tSum\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\t'
    'Z-Stack Group ID\tAcquire\tVideo Config\tStim_Config\tAuto_Named\n'
    'A2_BF\t1.0\t1.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\tA2\t\t-1\t'
    'False\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\tTrue\n'
    'Control_preDose\t2.0\t2.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\t'
    'A3\t\t-1\tFalse\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\tFalse\n'
    'Treatment_10x\t3.0\t3.0\t100.0\tFalse\tGreen\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\t'
    '\t\t-1\tTrue\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\tFalse\n'
    'custom0001_Red\t5.0\t5.0\t100.0\tFalse\tRed\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\t'
    '\t\t-1\tTrue\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\tTrue\n'
)


def test_v7_load_recovers_labels_and_saves_as_v8(tmp_path):
    src = tmp_path / 'proto_v7.tsv'
    src.write_text(_V7_TSV)
    proto = Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)
    steps = proto.steps()

    # The renamed step's user text is recovered verbatim into Label; auto
    # names recover their machine base ('' for a well anchor, the accreted
    # prefix for a custom step).
    assert list(steps['Label']) == ['', 'Control_preDose', 'Treatment_10x', 'custom0001']
    # Names re-render from the columns: byte-identical for auto rows, and
    # user labels gain their channel token.
    assert list(steps['Name']) == [
        'A2_BF',
        'Control_preDose_BF',
        'Treatment_10x_Green',
        'custom0001_Red',
    ]

    # Save -> the file is stamped Version 8; reload -> byte-identical Names
    # and Labels (the migration is a fixed point).
    dst = tmp_path / 'proto_v8.tsv'
    assert proto.to_file(file_path=dst) is None
    lines = dst.read_text().splitlines()
    assert lines[1] == 'Version\t8', lines[1]

    proto2 = Protocol.from_file(file_path=dst, tiling_configs_file_loc=TILING_CONFIGS)
    assert list(proto2.steps()['Name']) == list(steps['Name'])
    assert list(proto2.steps()['Label']) == list(steps['Label'])


_V5_TSV = (
    'LumaViewPro Protocol\n'
    'Version\t5\n'
    'Period\t0\n'
    'Duration\t0\n'
    'Labware\tCenter Plate\n'
    'Capture Root\t\n'
    'Steps\n'
    'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\tAuto_Gain\t'
    'Exposure\tSum\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\t'
    'Z-Stack Group ID\tAcquire\tVideo Config\tStim_Config\n'
    'A2_BF\t1.0\t1.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\tA2\t\t-1\t'
    'False\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\n'
    'B4_Green_TA1\t2.0\t2.0\t100.0\tFalse\tGreen\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\t'
    'B4\tA1\t-1\tFalse\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\n'
    'MyPickedSpot\t3.0\t3.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\tB4\t'
    '\t-1\tFalse\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\n'
)


def test_pre_v7_compose_and_compare_recovers_flag_per_row(tmp_path):
    src = tmp_path / 'proto_v5.tsv'
    src.write_text(_V5_TSV)
    proto = Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)
    steps = proto.steps()

    # Pure auto names reload byte-identically and are flagged auto=True (the
    # old back-fill marked every pre-v7 step user-named, freezing auto names
    # against channel changes forever).
    assert list(steps['Name']) == ['A2_BF', 'B4_Green_TA1', 'MyPickedSpot_BF']
    assert list(steps['Label']) == ['', '', 'MyPickedSpot']
    assert list(steps['Auto_Named']) == [True, True, False]


# ---------------------------------------------------------------------------
# T5: tiling / z-stack expansion children inherit the parent label.
# ---------------------------------------------------------------------------

_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}
_WIDE_Z = {'Z': {'limits': {'min': 0.0, 'max': 10000.0}}}


def _labeled_step(label='Treatment_10x', **kwargs):
    return _make_step(
        name=f'{label}_BF',
        label=label,
        auto_named=False,
        well='',
        z_slice=-1,
        tile='',
        tile_group_id=-1,
        zstack_group_id=-1,
        **kwargs,
    )


def test_zstack_children_keep_parent_label():
    proto = _build_protocol([_labeled_step(z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=_WIDE_Z)
    steps = proto.steps()
    assert len(steps) == 6
    assert list(steps['Label']) == ['Treatment_10x'] * 6
    for i, name in enumerate(steps['Name']):
        assert name == f'Treatment_10x_BF_Z{i}', name


def test_tiling_children_keep_parent_label(scale_ctx):
    from modules.labware_loader import WellPlateLoader

    labware = WellPlateLoader().get_plate('6 well microplate')
    axes_config = {
        'X': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
        'Y': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
    }
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    status = proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        curr_step_idx=0,
        axes_config=axes_config,
        labware=labware,
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
    )
    assert status['tiles_skipped'] == 0
    steps = proto.steps()
    assert len(steps) == 4, f'2x2 tiling must expand to 4 steps, got {len(steps)}'
    assert list(steps['Label']) == ['Treatment_10x'] * 4
    tiles = list(steps['Tile'])
    assert len(set(tiles)) == 4
    for name, tile in zip(steps['Name'], tiles, strict=True):
        assert name == f'Treatment_10x_BF_T{tile}', name


# ---------------------------------------------------------------------------
# T7: antifragility -- a label whose embedded segment ALREADY classifies as a
# token in the parse vocabulary still round-trips columns -> filename whole,
# proving labels live outside the vocabulary's blast radius.
# ---------------------------------------------------------------------------


def test_token_shaped_label_segments_survive_whole():
    for label in ('Assay_Turret3', 'Foo_stitched', 'Plate_T3', 'Run_Z12', 'Deck_0003'):
        base = _writer_base(_step_row(label, color='Green'))
        assert base == f'{label}_Green_0000', (
            f'label {label!r} must reach the filename unparsed; got {base!r}'
        )


def test_token_shaped_label_survives_protocol_roundtrip(tmp_path):
    proto = _build_protocol(
        [
            _make_step(
                name='Assay_Turret3_BF',
                label='Assay_Turret3',
                auto_named=False,
                well='',
                z_slice=-1,
                tile_group_id=-1,
                zstack_group_id=-1,
            )
        ]
    )
    path = tmp_path / 'p.tsv'
    assert proto.to_file(file_path=path) is None
    reloaded = Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS)
    step = reloaded.step(idx=0)
    assert step['Label'] == 'Assay_Turret3'
    assert step['Name'] == 'Assay_Turret3_BF'
    assert _writer_base(step) == 'Assay_Turret3_BF_0000'


# ---------------------------------------------------------------------------
# Post-record: a v1 file loads, back-fills Label, and is rewritten in place
# as v2 so appended rows cannot misalign under the old header.
# ---------------------------------------------------------------------------

_POST_V1_HEADER = (
    'LumaViewPro Protocol Post-Processing Record\n'
    'Version\t1\n'
    '\n'
    'Images\n'
    'Filepath\tTimestamp\tName\tScan Count\tX\tY\tZ\tZ-Slice\tWell\tColor\tObjective\t'
    'Tile Group ID\tTile\tCustom Step\tComposite\tStitched\tZProject\tVideo\tHyperstack\n'
)

_POST_V1_ROWS = (
    'img1.tiff\t2026-06-01 10:00:00\tA2_BF\t0\t1.0\t2.0\t3.0\t-1\tA2\tBF\t4x\t-1\t\t'
    'False\tFalse\tFalse\tFalse\tFalse\tFalse\n'
    'img2.tiff\t2026-06-01 10:00:01\tTreatment_10x\t0\t1.0\t2.0\t3.0\t-1\t\tGreen\t4x\t-1\t\t'
    'True\tFalse\tFalse\tFalse\tFalse\tFalse\n'
)


def test_post_record_v1_upgrades_in_place_and_appends_aligned(tmp_path):
    record_path = tmp_path / 'protocol_post_record.tsv'
    record_path.write_text(_POST_V1_HEADER + _POST_V1_ROWS)

    record = ProtocolPostRecord.from_file(file_path=record_path)
    df = record.records()
    assert len(df) == 2
    # Compose-and-compare back-fill: the auto name recovers the machine base
    # (''), the user text is kept verbatim.
    assert list(df['Label']) == ['', 'Treatment_10x']
    assert list(df['Name']) == ['A2_BF', 'Treatment_10x']

    # The on-disk file was rewritten as v2 with the Label column, rows intact.
    lines = record_path.read_text().splitlines()
    assert lines[1] == 'Version\t2', lines[1]
    header_idx = lines.index('Images') + 1
    columns = lines[header_idx].split('\t')
    assert columns == list(ProtocolPostRecord.COLUMNS)
    label_idx = columns.index('Label')
    rows = [line.split('\t') for line in lines[header_idx + 1 :] if line]
    assert len(rows) == 2
    assert [row[0] for row in rows] == ['img1.tiff', 'img2.tiff']
    assert [row[label_idx] for row in rows] == ['', 'Treatment_10x']

    # add_record on the upgraded instance appends a correctly-aligned row.
    record.add_record(
        root_path=tmp_path,
        file_path=pathlib.Path('stitched.tiff'),
        timestamp=datetime.datetime(2026, 6, 1, 10, 5, 0),
        name='Treatment_10x_Green_stitched',
        label='Treatment_10x',
        scan_count=0,
        x=1.0,
        y=2.0,
        z=3.0,
        z_slice=-1,
        well='',
        color='Green',
        objective='4x',
        tile_group_id=-1,
        tile='',
        custom_step=True,
        **dict.fromkeys(ProtocolPostRecord.COLUMNS[15:], False) | {'Stitched': True},
    )
    record.complete()

    lines = record_path.read_text().splitlines()
    rows = [line.split('\t') for line in lines[header_idx + 1 :] if line]
    assert len(rows) == 3
    appended = rows[2]
    assert len(appended) == len(columns), (
        f'appended row has {len(appended)} cells for {len(columns)} columns'
    )
    row_map = dict(zip(columns, appended, strict=True))
    assert row_map['Filepath'] == 'stitched.tiff'
    assert row_map['Label'] == 'Treatment_10x'
    assert row_map['Stitched'] == 'True'
    assert row_map['Custom Step'] == 'True'

    # And the reloaded (now-v2) file parses cleanly with the appended row.
    record2 = ProtocolPostRecord.from_file(file_path=record_path)
    df2 = record2.records()
    assert len(df2) == 3
    assert list(df2['Label']) == ['', 'Treatment_10x', 'Treatment_10x']
    record2.complete()


# ===========================================================================
# Review-fix regressions (post-review hardening of the Label rewire).
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1: recover_step_label's second machine shape -- a Name that is its own
# anchor followed only by vocabulary tokens is machine-generated, even when
# it no longer renders byte-equal (stale tokens from old channel-change bugs,
# or post-record rows whose columns were output-adjusted).
# ---------------------------------------------------------------------------


def test_recover_label_stale_token_name_is_machine_and_cleaned():
    # Old releases appended a channel token instead of replacing it; the row
    # must classify machine so the re-render CLEANS the stale token rather
    # than freezing the doubled name as a user label.
    row = {'Name': 'A2_BF_Green', 'Well': 'A2', 'Color': 'Green', 'Tile': '', 'Z-Slice': -1}
    assert recover_step_label(row) == ('', True)
    cleaned = {**row, 'Label': ''}
    assert build_step_name(step_components(cleaned)) == 'A2_Green'


def test_recover_label_user_text_embedding_anchor_survives():
    # User text that happens to start with the well anchor is NOT machine --
    # 'treated' does not classify as a token, so the text is kept verbatim.
    row = {'Name': 'A2_treated', 'Well': 'A2', 'Color': 'BF', 'Tile': '', 'Z-Slice': -1}
    assert recover_step_label(row) == ('A2_treated', False)


def test_recover_label_stale_token_custom_name_keeps_machine_prefix():
    row = {'Name': 'custom0001_BF_Green', 'Well': '', 'Color': 'Green', 'Tile': '', 'Z-Slice': -1}
    assert recover_step_label(row) == ('custom0001', True)


def test_recover_label_post_record_output_adjusted_rows_stay_machine():
    # A stitch blanks the Tile column while Name keeps the source tile token;
    # a composite rewrites Color to 'Composite' while Name keeps the source
    # channel. Both must classify machine (anchor + tokens), or chained
    # post-processing would treat the old name as a user label and double
    # its tokens on the next render.
    stitched = {'Name': 'B4_Green_TA1', 'Well': 'B4', 'Color': 'Green', 'Tile': '', 'Z-Slice': -1}
    assert recover_step_label(stitched) == ('', True)
    composite = {'Name': 'A2_BF', 'Well': 'A2', 'Color': 'Composite', 'Tile': '', 'Z-Slice': -1}
    assert recover_step_label(composite) == ('', True)


# ---------------------------------------------------------------------------
# Fix 2: a blank Z-Slice cell forces the column to float64 at read; the
# loader must normalize to the int / -1-sentinel form so compose-and-compare
# still matches ('Z3' not 'Z3.0') and names never render float tokens.
# ---------------------------------------------------------------------------

_V6_ZFLOAT = (
    'LumaViewPro Protocol\n'
    'Version\t6\n'
    'Period\t0\n'
    'Duration\t0\n'
    'Labware\tCenter Plate\n'
    'Capture Root\t\n'
    'Steps\n'
    'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\tAuto_Gain\t'
    'Exposure\tSum\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\t'
    'Z-Stack Group ID\tAcquire\tVideo Config\tStim_Config\n'
    'A2_BF_Z3\t1.0\t1.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\tA2\t\t3\t'
    'False\t-1\t0\timage\t{"fps": 5, "duration": 5}\t{}\n'
    'A3_BF\t2.0\t2.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\tA3\t\t\t'
    'False\t-1\t-1\timage\t{"fps": 5, "duration": 5}\t{}\n'
)


def test_float_zslice_column_does_not_misclassify_names(tmp_path):
    src = tmp_path / 'v6_zfloat.tsv'
    src.write_text(_V6_ZFLOAT)
    proto = Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)
    steps = proto.steps()
    # The Z3 row must still compose-and-compare as machine (Label '') and
    # re-render byte-identically -- a float 3.0 would have rendered 'Z3.0',
    # failed the compare, and frozen 'A2_BF_Z3' as a user label.
    assert list(steps['Name']) == ['A2_BF_Z3', 'A3_BF']
    assert list(steps['Label']) == ['', '']
    assert list(steps['Z-Slice']) == [3, -1]


# ---------------------------------------------------------------------------
# Fix 3: the oldest files name the channel column 'Channel'; the loader must
# normalize it to 'Color' instead of KeyError-ing in name derivation. And
# Well / Tile / Z-Slice are load-time REQUIREMENTS now that names derive
# from them -- a missing column fails loud at validation, not as a KeyError
# deep in the migration.
# ---------------------------------------------------------------------------

_V2_CHANNEL = (
    'LumaViewPro Protocol\n'
    'Version\t2\n'
    'Period\t0\n'
    'Duration\t0\n'
    'Labware\tCenter Plate\n'
    'Capture Root\t\n'
    'Steps\n'
    'Name\tX\tY\tZ\tAuto_Focus\tChannel\tFalse_Color\tIllumination\tGain\tAuto_Gain\t'
    'Exposure\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\tZ-Stack Group ID\n'
    'A2_BF\t1.0\t1.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t4x\tA2\t\t-1\t'
    'False\t-1\t-1\n'
)


def test_legacy_channel_column_loads_and_normalizes_to_color(tmp_path):
    src = tmp_path / 'v2_channel.tsv'
    src.write_text(_V2_CHANNEL)
    proto = Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)
    steps = proto.steps()
    assert list(steps['Name']) == ['A2_BF']
    assert 'Color' in steps.columns
    assert steps.iloc[0]['Color'] == 'BF'
    assert steps.iloc[0]['Label'] == ''


def test_missing_well_column_raises_format_error(tmp_path):
    # Drop the Well column entirely: the loader must refuse with a
    # ProtocolFormatError naming the missing column, not KeyError later.
    src = tmp_path / 'v2_no_well.tsv'
    src.write_text(_V2_CHANNEL.replace('\tWell', '').replace('4x\tA2\t', '4x\t'))
    with pytest.raises(ProtocolFormatError, match='Well'):
        Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)


# ---------------------------------------------------------------------------
# Fix 4: a numeric-looking label ('0600') must survive save/reload as the
# string it was typed as, not pandas-inferred into a float/int.
# ---------------------------------------------------------------------------


def test_numeric_looking_label_survives_protocol_roundtrip(tmp_path):
    src = tmp_path / 'v7.tsv'
    src.write_text(_V7_TSV)
    proto = Protocol.from_file(file_path=src, tiling_configs_file_loc=TILING_CONFIGS)
    proto.modify_name(step_idx=0, step_name='0600')
    assert proto.step(idx=0)['Label'] == '0600'

    dst = tmp_path / 'v8.tsv'
    assert proto.to_file(file_path=dst) is None
    proto2 = Protocol.from_file(file_path=dst, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto2.step(idx=0)['Label'] == '0600'
    assert proto2.step(idx=0)['Name'] == '0600_BF'


def test_numeric_looking_label_survives_post_record_reload(tmp_path):
    record_path = tmp_path / 'protocol_post_record.tsv'
    header = (
        'LumaViewPro Protocol Post-Processing Record\n'
        'Version\t2\n'
        '\n'
        'Images\n' + '\t'.join(ProtocolPostRecord.COLUMNS) + '\n'
    )
    rows = (
        'img1.tiff\t2026-06-01 10:00:00\t0600_BF\t0600\t0\t1.0\t2.0\t3.0\t-1\t\tBF\t4x\t-1\t\t'
        'True\tFalse\tFalse\tFalse\tFalse\tFalse\n'
        'img2.tiff\t2026-06-01 10:00:01\tA2_BF\t\t0\t1.0\t2.0\t3.0\t-1\tA2\tBF\t4x\t-1\t\t'
        'False\tFalse\tFalse\tFalse\tFalse\tFalse\n'
    )
    record_path.write_text(header + rows)
    record = ProtocolPostRecord.from_file(file_path=record_path)
    df = record.records()
    assert list(df['Label']) == ['0600', '']
    assert list(df['Name']) == ['0600_BF', 'A2_BF']
    record.complete()


# ---------------------------------------------------------------------------
# Fix 5: a rename that sanitizes to nothing is refused at every entry point
# (shared Protocol._sanitized_label); silently storing '' would revert the
# step to its machine base and collapse same-channel custom steps onto one
# filename.
# ---------------------------------------------------------------------------


def _labeled_single_step_protocol():
    return _build_protocol(
        [
            _make_step(
                name='MySpot_BF',
                label='MySpot',
                auto_named=False,
                well='',
                z_slice=-1,
                tile_group_id=-1,
                zstack_group_id=-1,
            )
        ]
    )


def test_modify_name_refuses_name_that_sanitizes_to_empty():
    proto = _labeled_single_step_protocol()
    with pytest.raises(ProtocolError):
        proto.modify_name(step_idx=0, step_name='###')
    assert proto.step(idx=0)['Label'] == 'MySpot'
    assert proto.step(idx=0)['Name'] == 'MySpot_BF'


def test_modify_step_refuses_label_that_sanitizes_to_empty():
    proto = _labeled_single_step_protocol()
    layer_config = {
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 100.0,
        'gain_db': 10.0,
        'auto_gain': False,
        'exposure_ms': 5.0,
        'sum': 1,
        'acquire': 'image',
        'video_config': {'duration': 5, 'fps': 30},
    }
    with pytest.raises(ProtocolError):
        proto.modify_step(
            step_idx=0,
            label='###',
            layer='Green',
            layer_config=layer_config,
            plate_position={'x': 0.0, 'y': 0.0, 'z': 5000.0},
            objective_id='4x Oly',
            stim_configs={},
        )
    step = proto.step(idx=0)
    assert step['Label'] == 'MySpot'
    assert step['Name'] == 'MySpot_BF'
    assert step['Color'] == 'BF', 'refused rename must leave the step unmodified'


def test_insert_step_refuses_name_that_sanitizes_to_empty():
    proto = _labeled_single_step_protocol()
    layer_config = {
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 100.0,
        'gain_db': 10.0,
        'auto_gain': False,
        'exposure_ms': 5.0,
        'sum': 1,
        'acquire': 'image',
        'video_config': {'duration': 5, 'fps': 30},
    }
    with pytest.raises(ProtocolError):
        proto.insert_step(
            step_name='###',
            layer='BF',
            layer_config=layer_config,
            plate_position={'x': 0.0, 'y': 0.0, 'z': 5000.0},
            objective_id='4x Oly',
            stim_configs={},
            before_step=0,
            after_step=None,
        )
    assert proto.num_steps() == 1, 'refused insert must not add a step'


# ---------------------------------------------------------------------------
# Fix 6: a header-only (zero-row) v1 post-record is a legitimate state --
# it loads, upgrades in place to v2 with the Label column, and stays
# appendable.
# ---------------------------------------------------------------------------

_POST_V1_EMPTY = (
    'LumaViewPro Protocol Post-Processing Record\n'
    'Version\t1\n'
    '\n'
    'Images\n'
    'Filepath\tTimestamp\tName\tScan Count\tX\tY\tZ\tZ-Slice\tWell\tColor\tObjective\t'
    'Tile Group ID\tTile\tCustom Step\tComposite\tStitched\tZProject\tVideo\tHyperstack\n'
)


def test_header_only_v1_post_record_loads_and_upgrades(tmp_path):
    record_path = tmp_path / 'protocol_post_record.tsv'
    record_path.write_text(_POST_V1_EMPTY)

    record = ProtocolPostRecord.from_file(file_path=record_path)
    assert len(record.records()) == 0

    lines = record_path.read_text().splitlines()
    assert lines[1] == 'Version\t2', lines[1]
    header_idx = lines.index('Images') + 1
    columns = lines[header_idx].split('\t')
    assert columns == list(ProtocolPostRecord.COLUMNS)
    assert 'Label' in columns

    # The upgraded empty record accepts an aligned append.
    record.add_record(
        root_path=tmp_path,
        file_path=pathlib.Path('out.tiff'),
        timestamp=datetime.datetime(2026, 6, 1, 12, 0, 0),
        name='A2_BF_stitched',
        label='',
        scan_count=0,
        x=1.0,
        y=2.0,
        z=3.0,
        z_slice=-1,
        well='A2',
        color='BF',
        objective='4x',
        tile_group_id=-1,
        tile='',
        custom_step=False,
        **dict.fromkeys(ProtocolPostRecord.COLUMNS[15:], False) | {'Stitched': True},
    )
    record.complete()

    record2 = ProtocolPostRecord.from_file(file_path=record_path)
    df = record2.records()
    assert len(df) == 1
    assert df.iloc[0]['Label'] == ''
    assert str(df.iloc[0]['Filepath']) == 'out.tiff'
    assert bool(df.iloc[0]['Stitched']) is True
    record2.complete()


# ---------------------------------------------------------------------------
# Fix 7: the v1 -> v2 rewrite is atomic -- an exception mid-rewrite leaves
# the original file byte-identical and removes the temp file.
# ---------------------------------------------------------------------------


def test_rewrite_failure_preserves_original_and_removes_tmp(tmp_path, monkeypatch):
    record_path = tmp_path / 'protocol_post_record.tsv'
    original_text = _POST_V1_HEADER + _POST_V1_ROWS
    record_path.write_text(original_text)

    def _boom(self, **kwargs):
        raise RuntimeError('disk full mid-rewrite')

    monkeypatch.setattr(ProtocolPostRecord, '_add_record_to_file', _boom)
    with pytest.raises(RuntimeError, match='disk full mid-rewrite'):
        ProtocolPostRecord.from_file(file_path=record_path)

    assert record_path.read_text() == original_text, (
        'a failed rewrite must leave the original v1 file untouched'
    )
    tmp_file = record_path.with_name(record_path.name + '.tmp')
    assert not tmp_file.exists(), 'the aborted rewrite must remove its temp file'


# ---------------------------------------------------------------------------
# Fix 8: an unreadable post-record file is moved aside as <name>.unreadable
# and a fresh current-version record starts in its place -- never appended
# to under its old header.
# ---------------------------------------------------------------------------


def test_unreadable_post_record_moved_aside_and_fresh_record_started(tmp_path, monkeypatch):
    import modules.protocol_post_processing_helper as pph
    from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

    corrupt_path = tmp_path / 'protocol_post_record.tsv'
    corrupt_text = 'THIS IS NOT A POST RECORD\ngarbage\n'
    corrupt_path.write_text(corrupt_text)

    helper = ProtocolPostProcessingHelper()
    monkeypatch.setattr(
        helper,
        '_find_protocol_tsvs',
        lambda path: {
            'protocol_root_dir': tmp_path,
            'protocol': tmp_path / 'protocol.tsv',
            'protocol_execution_record': tmp_path / 'protocol_execution_record.tsv',
            'protocol_post_record': corrupt_path,
        },
    )
    monkeypatch.setattr(pph.Protocol, 'from_file', staticmethod(lambda **kwargs: MagicMock()))
    exec_record = MagicMock()
    exec_record.num_records.return_value = 1
    monkeypatch.setattr(
        pph.ProtocolExecutionRecord, 'from_file', staticmethod(lambda **kwargs: exec_record)
    )
    monkeypatch.setattr(
        helper, '_get_image_filenames_from_folder', lambda **kwargs: {'raw': [], 'post': []}
    )
    monkeypatch.setattr(helper, '_get_raw_images_df', lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(helper, '_get_post_images_df', lambda **kwargs: pd.DataFrame())

    result = helper.load_folder(path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS)

    # The unreadable file is preserved byte-for-byte under .unreadable.
    preserved = corrupt_path.with_name(corrupt_path.name + '.unreadable')
    assert preserved.exists()
    assert preserved.read_text() == corrupt_text

    # A fresh current-version record replaced it at the default location.
    fresh_lines = corrupt_path.read_text().splitlines()
    assert fresh_lines[0] == ProtocolPostRecord.FILE_HEADER
    assert fresh_lines[1] == 'Version\t2'
    header_idx = fresh_lines.index('Images') + 1
    assert fresh_lines[header_idx].split('\t') == list(ProtocolPostRecord.COLUMNS)

    # No images in the folder, so the load reports that -- but it did NOT
    # fail on (or append to) the unreadable record.
    assert result['status'] is False
    assert 'No image files' in result['message']


# ---------------------------------------------------------------------------
# Fix 9: v1 post-record migration of derived-output rows (stitched et al.) --
# the anchor-shape test keeps their machine names out of Label so chained
# post-processing cannot double tokens.
# ---------------------------------------------------------------------------


def test_v1_post_record_stitched_row_backfills_empty_label(tmp_path):
    record_path = tmp_path / 'protocol_post_record.tsv'
    rows = (
        # A stitched output: Tile blanked by the stitch, Name keeps the
        # source tile token.
        'B4_Green_TA1_stitched.tiff\t2026-06-01 10:00:00\tB4_Green_TA1\t0\t1.0\t2.0\t3.0\t-1\t'
        'B4\tGreen\t4x\t-1\t\tFalse\tFalse\tTrue\tFalse\tFalse\tFalse\n'
        # A user-labeled source row rides along verbatim.
        'treated.tiff\t2026-06-01 10:00:01\tTreatment_10x\t0\t1.0\t2.0\t3.0\t-1\t\tGreen\t4x\t'
        '-1\t\tTrue\tFalse\tFalse\tFalse\tFalse\tFalse\n'
    )
    record_path.write_text(_POST_V1_HEADER + rows)

    record = ProtocolPostRecord.from_file(file_path=record_path)
    df = record.records()
    assert list(df['Label']) == ['', 'Treatment_10x'], (
        'a stitched row must recover the machine base, not its full stale name'
    )
    # The derived output base for the stitched row renders from columns
    # (tile already blanked) -- no doubled tile token.
    base = build_step_name(step_components(df.iloc[0], post=('stitched',)))
    assert base == 'B4_Green_stitched', base
    record.complete()
