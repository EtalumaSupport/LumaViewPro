# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Collision-loud policy: identical derived capture/output filenames refuse
loudly instead of leaning on write-time rename suffixes.

Layers covered:
- validate_for_run refuses a run whose steps render duplicate capture bases
  (T6); an image + video step sharing a Name is NOT a collision (the video
  renders its '_video' token), and neither are steps differing only in
  Objective (the writer stamps the objective per step on turret builds).
- Load WARNS (never rejects -- rejection would block the in-app rename that
  is the remedy) on any rendered-base collision; the run itself is refused
  at start, which is the data-loss gate.
- Labels are re-sanitized loudly at the load boundary so hand-edited files
  cannot smuggle in labels that differ only in write-time-stripped chars.
- custom_step_count resumes past the highest 'custom<NNNN>' label in a
  loaded file, so a new added step cannot collide with an existing one.
- Post-processor load_folder pre-scans planned output names and refuses
  exactly the colliding groups (per-group), still generating the clean ones;
  resume skips are unaffected.
- VideoWriter resolves collisions at encoder init (so the cv2 .avi rewrite
  cannot dodge the check) and `output_path` is the record authority.
- Tiling inference reads the Tile column, never step names, so tile-shaped
  user text cannot fake a tiling; malformed Tile cells are skipped loudly.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pandas as pd

from modules.common_utils import PostFunction
from modules.protocol import Protocol
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult

from tests.test_protocol_overwrite_guard import _build_tsv, _step_row
from tests.test_validate_steps import (
    _DEFAULT_AXIS_LIMITS,
    _STAGE_OFFSET,
    _make_protocol,
    _valid_step,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


# ---------------------------------------------------------------------------
# validate_for_run: the refuse-at-run-start collision policy (T6).
# ---------------------------------------------------------------------------


def test_validate_for_run_refuses_two_steps_renamed_to_one_label():
    # Two well steps renamed to the same label render one capture base;
    # the run must refuse before any hardware moves, naming both steps
    # and the colliding base.
    p = _make_protocol(
        [
            _valid_step(
                Name='Control_Blue_Z0',
                Label='Control',
                Well='A1',
                X=60.0,
                Y=40.0,
                Z=5000.0,
                **{'Tile Group ID': 0},
            ),
            _valid_step(
                Name='Control_Blue_Z0',
                Label='Control',
                Well='A2',
                X=61.0,
                Y=40.0,
                Z=5000.0,
                **{'Tile Group ID': 1},
            ),
        ]
    )
    errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS, stage_offset=_STAGE_OFFSET)
    collision_errors = [e for e in errors if 'would save captures' in e]
    assert len(collision_errors) == 1, errors
    msg = collision_errors[0]
    assert 'Steps 1, 2' in msg
    assert "'Control_Blue_Z0_...'" in msg
    assert 'Rename' in msg


def test_validate_for_run_allows_image_and_video_step_sharing_name():
    # A video step renders with its 'video' post token, so an image step
    # and a video step may legitimately share a Name without colliding.
    video_cfg = {'duration': 5.0, 'fps': 5}
    p = _make_protocol(
        [
            _valid_step(Well='A1', X=60.0, Y=40.0, Z=5000.0, Acquire='image'),
            _valid_step(
                Well='A1',
                X=60.0,
                Y=40.0,
                Z=5000.0,
                Acquire='video',
                **{'Video Config': video_cfg},
            ),
        ]
    )
    errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS, stage_offset=_STAGE_OFFSET)
    assert not any('would save captures' in e for e in errors), errors


# ---------------------------------------------------------------------------
# Load-time checks, re-keyed on the RENDERED capture base.
# ---------------------------------------------------------------------------


def test_load_warns_same_base_in_same_tile_group_and_still_loads(tmp_path, monkeypatch):
    # Two steps on DIFFERENT wells renamed to one label render the same
    # base. The load must NOT reject -- a load-time rejection would block
    # the in-app rename that is the remedy -- it warns once, and the run
    # itself is refused at start (validate_for_run), the data-loss gate.
    from modules import protocol as protocol_mod

    captured: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            captured.append(message)

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    rows = ''
    rows += _step_row('Control', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('Control', 'A2', '', -1, 0, 60.1, 34.6, 5001.7)
    tsv = tmp_path / 'renamed_dup.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto.num_steps() == 2, 'the file must load so the user can rename the steps'
    assert len(captured) == 1, captured
    assert 'refused' in captured[0].lower()
    assert 'rename' in captured[0].lower()


def test_load_soft_warns_same_base_across_tile_groups(tmp_path, monkeypatch):
    from modules import protocol as protocol_mod

    captured: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            captured.append((category, title, message))

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    rows = ''
    rows += _step_row('Control', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('Control', 'A2', '', -1, 1, 60.1, 34.6, 5001.7)
    tsv = tmp_path / 'renamed_cross_tgid.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto.num_steps() == 2, 'the file must still load so the user can edit the names'
    assert len(captured) == 1, captured
    _category, _title, message = captured[0]
    assert 'refused' in message.lower(), (
        'the warning must say the run will be refused, not promise a rename suffix'
    )


# ---------------------------------------------------------------------------
# custom_step_count resumes past the highest loaded 'custom<NNNN>' label.
# ---------------------------------------------------------------------------


def _insert_layer_config():
    return {
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


def test_custom_step_count_resumes_past_loaded_labels(tmp_path):
    rows = _step_row('custom0007_BF', '', '', -1, -1, 46.5, 34.6, 4972.9)
    tsv = tmp_path / 'custom7.tsv'
    tsv.write_text(_build_tsv(rows))
    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto.step(idx=0)['Label'] == 'custom0007'

    returned_name = proto.insert_step(
        step_name=None,
        layer='BF',
        layer_config=_insert_layer_config(),
        plate_position={'x': 10.0, 'y': 10.0, 'z': 5000.0},
        objective_id='4x Oly',
        stim_configs={},
        before_step=None,
        after_step=0,
    )
    # A reset-to-zero counter would mint 'custom0000'; a counter that
    # resumed at the loaded step would re-mint 'custom0007' and collide.
    assert returned_name == 'custom0008_BF', returned_name
    assert proto.step(idx=1)['Label'] == 'custom0008'


# ---------------------------------------------------------------------------
# Post-processor output pre-scan: cross-group collisions refuse everything;
# resume skips (already-recorded outputs) keep their behavior.
# ---------------------------------------------------------------------------


class _FakePostProcessor(ProtocolPostProcessor):
    """Minimal concrete subclass driving the BASE class load_folder logic."""

    def __init__(self):
        super().__init__(post_function=PostFunction.ZPROJECT, has_turret=False)
        self.algorithm_calls: list = []
        self.records_added: list = []

    def _get_groups(self, df):
        return [(key, group) for key, group in df.groupby('GroupKey')]

    def _generate_filename(self, df, **kwargs):
        return df.iloc[0]['OutName']

    def _filter_ignored_types(self, df):
        return df

    def _group_algorithm(self, path, df, **kwargs):
        self.algorithm_calls.append(df.iloc[0]['OutName'])
        actual = df.iloc[0].get('ActualOutPath')
        # A writer that relocated the output (collision suffix, container
        # fallback) reports where the file really landed.
        return PostProcResult.ok(
            significant_bits=8,
            actual_output_file_loc=actual if actual else None,
        )

    def _add_record(self, protocol_post_record, alg_metadata, root_path, **kwargs):
        self.records_added.append(kwargs.get('file_path'))


def _fake_images_df(group_out_names):
    rows = []
    for group_key, out_name in enumerate(group_out_names):
        for i in range(2):  # two frames per group so the group is eligible
            row = {
                'Filepath': f'g{group_key}_f{i}.tiff',
                'GroupKey': group_key,
                'OutName': out_name,
            }
            row.update(dict.fromkeys(PostFunction.list_values(), False))
            rows.append(row)
    return pd.DataFrame(rows)


def _drive_load_folder(processor, tmp_path, monkeypatch, images_df, post_record):
    monkeypatch.setattr(
        processor._post_processing_helper,
        'load_folder',
        lambda **kwargs: {
            'status': True,
            'images_df': images_df,
            'root_path': tmp_path,
            'protocol_post_record': post_record,
            'protocol': None,
        },
    )
    return processor.load_folder(path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS)


def test_post_processor_all_groups_colliding_refuses_with_reason(tmp_path, monkeypatch):
    # Every eligible group renders one output name -> nothing can be
    # generated; the whole operation reports the collision as its reason.
    processor = _FakePostProcessor()
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    images_df = _fake_images_df(['A1_BF_zproj.tiff', 'A1_BF_zproj.tiff'])

    result = _drive_load_folder(processor, tmp_path, monkeypatch, images_df, post_record)

    assert result['status'] is False
    assert result['reason'] == 'collision'
    assert 'No ZProject was generated' in result['message']
    assert processor.algorithm_calls == [], 'nothing may be generated when every group collides'
    assert processor.records_added == []


def test_post_processor_mixed_collision_refuses_only_colliding_groups(tmp_path, monkeypatch):
    # Per-group refusal: the colliding pair is refused (their artifact
    # would be indistinguishable) while the clean group still generates
    # and is recorded. The operation succeeds with a note naming the
    # refusal so an already-captured folder stays post-processable.
    processor = _FakePostProcessor()
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    images_df = _fake_images_df(['A1_BF_zproj.tiff', 'A1_BF_zproj.tiff', 'B2_Green_zproj.tiff'])

    result = _drive_load_folder(processor, tmp_path, monkeypatch, images_df, post_record)

    assert result['status'] is True
    assert result['message'].startswith('Success.')
    assert 'refused' in result['message']
    assert 'A1_BF_zproj.tiff' in result['message']
    assert processor.algorithm_calls == ['B2_Green_zproj.tiff'], 'only the clean group may generate'
    assert len(processor.records_added) == 1
    assert str(processor.records_added[0]).endswith('B2_Green_zproj.tiff')
    post_record.complete.assert_called_once()


def test_post_processor_records_writer_relocated_output(tmp_path, monkeypatch):
    # When the group algorithm reports the output landed elsewhere
    # (collision suffix / container fallback), the record must carry the
    # ACTUAL path, not the requested one.
    processor = _FakePostProcessor()
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    images_df = _fake_images_df(['A1_BF_video.mp4'])
    actual = tmp_path / 'ZProject' / 'A1_BF_video_000001.avi'
    images_df['ActualOutPath'] = str(actual)

    result = _drive_load_folder(processor, tmp_path, monkeypatch, images_df, post_record)

    assert result['status'] is True
    assert len(processor.records_added) == 1
    recorded = pathlib.Path(str(processor.records_added[0]))
    assert recorded == actual.relative_to(tmp_path), (
        f'the record must point at the relocated file; got {recorded}'
    )


def test_post_processor_resume_skip_unaffected_by_collision_check(tmp_path, monkeypatch):
    processor = _FakePostProcessor()
    post_record = MagicMock()
    # Distinct names, both already in the record from a previous session.
    post_record.file_exists_in_records.return_value = True
    images_df = _fake_images_df(['A1_BF_zproj.tiff', 'A2_BF_zproj.tiff'])

    result = _drive_load_folder(processor, tmp_path, monkeypatch, images_df, post_record)

    assert result == {
        'status': True,
        'message': 'Success.',
        'new_count': 0,
        'output_root': str(tmp_path),
        'accounting_note': '',
    }
    assert processor.algorithm_calls == [], 'already-recorded outputs are skipped, not regenerated'
    post_record.complete.assert_called_once()


# ---------------------------------------------------------------------------
# VideoWriter: never silently overwrite an existing output.
# ---------------------------------------------------------------------------


def _patch_video_writer_logger(monkeypatch):
    from modules import video_writer as vw_mod

    captured: list = []

    class _RecordingLogger:
        def warning(self, msg, *args, **kwargs):
            captured.append(('WARNING', msg % args if args else msg))

        def info(self, msg, *args, **kwargs):
            captured.append(('INFO', msg % args if args else msg))

    monkeypatch.setattr(vw_mod, 'logger', _RecordingLogger())
    return captured


def _write_one_frame_and_close(writer):
    import datetime

    import numpy as np

    writer.add_frame(
        image=np.zeros((32, 32), dtype=np.uint8),
        timestamp=datetime.datetime(2026, 7, 1, 12, 0, 0),
    )
    writer.close()


def test_video_writer_uniquifies_existing_output_path(tmp_path, monkeypatch):
    # Collision resolution runs at encoder init -- after the backend (and
    # so the REAL container suffix) is known -- and writer.output_path is
    # the record authority. Pre-create the path the active backend will
    # actually open (.avi covers the cv2 fallback; .mp4 covers PyAV).
    from modules.video_writer import VideoWriter

    captured = _patch_video_writer_logger(monkeypatch)
    requested = tmp_path / 'A1_BF_video.mp4'
    for suffix in ('.mp4', '.avi'):
        requested.with_suffix(suffix).write_bytes(b'previous recording')

    writer = VideoWriter(output_path=requested, fps=5.0)
    _write_one_frame_and_close(writer)

    actual = writer.output_path
    assert actual.stem == 'A1_BF_video_000001', (
        f'an existing output must gain a numeric suffix; writer landed on {actual}'
    )
    assert actual.exists() and actual.stat().st_size > 0
    for suffix in ('.mp4', '.avi'):
        assert requested.with_suffix(suffix).read_bytes() == b'previous recording', (
            'the pre-existing recording must be untouched'
        )
    assert any(level == 'WARNING' and 'collision' in msg for level, msg in captured), (
        f'the rename must be surfaced as a warning; captured: {captured}'
    )


def test_video_writer_keeps_fresh_output_path(tmp_path, monkeypatch):
    from modules.video_writer import VideoWriter

    captured = _patch_video_writer_logger(monkeypatch)
    requested = tmp_path / 'A1_BF_video.mp4'

    writer = VideoWriter(output_path=requested, fps=5.0)
    _write_one_frame_and_close(writer)

    actual = writer.output_path
    # The backend may rewrite the container suffix (.avi fallback), but a
    # fresh path never gains a collision suffix.
    assert actual.stem == 'A1_BF_video', f'no collision -> the plain name; got {actual.name}'
    assert actual.exists()
    assert not any('collision' in msg for _, msg in captured)


def test_write_video_records_writers_actual_path(tmp_path, monkeypatch):
    # The protocol video write path must return the file that exists on
    # disk -- the writer's path -- not the requested name, when the
    # requested output is already taken.
    import datetime
    import queue as _queue

    import numpy as np

    from modules.video_capture import VideoCaptureResult, write_video

    frames = _queue.Queue()
    for i in range(2):
        frames.put((np.zeros((32, 32), dtype=np.uint8), datetime.datetime(2026, 7, 1, 12, 0, i)))

    # Pre-create both container suffixes so whichever backend is active
    # collides with the requested name.
    for suffix in ('.mp4', '.avi'):
        (tmp_path / 'clip').with_suffix(suffix).write_bytes(b'taken')

    result = VideoCaptureResult(
        captured_frames=2,
        calculated_fps=5,
        video_images=frames,
        duration_sec=0.4,
        dropped_frames=0,
    )
    capture_result = write_video(
        result=result,
        save_folder=tmp_path,
        name='clip',
        video_as_frames=False,
        step={'Color': 'BF', 'False_Color': False},
        callbacks={},
        save_encoding='8bit',
        capture_depth=8,
    )

    assert capture_result is not None
    assert capture_result.stem == 'clip_000001', (
        f'write_video must report the suffixed file the writer landed on; got {capture_result}'
    )
    assert capture_result.exists() and capture_result.stat().st_size > 0
    for suffix in ('.mp4', '.avi'):
        assert (tmp_path / 'clip').with_suffix(suffix).read_bytes() == b'taken'


def test_video_builder_create_video_reports_actual_output_file(tmp_path):
    # _create_video reports where the file really landed so the base class
    # records the artifact that exists, not the requested name.
    import numpy as np
    import tifffile as tf

    from modules.video_builder import VideoBuilder

    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    for i in range(2):
        tf.imwrite(str(frames_dir / f'frame_{i:04}.tiff'), np.full((32, 32), 128, dtype=np.uint8))
    df = pd.DataFrame(
        {
            'Filepath': [f'frame_{i:04}.tiff' for i in range(2)],
            'Scan Count': range(2),
            'Timestamp': '',
            'Color': None,
        }
    )

    requested = tmp_path / 'out.mp4'
    for suffix in ('.mp4', '.avi'):
        requested.with_suffix(suffix).write_bytes(b'taken')

    builder = VideoBuilder(has_turret=False)
    result = builder._create_video(
        path=frames_dir,
        df=df,
        frames_per_sec=5,
        enable_timestamp_overlay=False,
        output_file_loc=requested,
        popup=None,
        total_groups=1,
        current_group=1,
    )
    assert result['status'] is True
    actual = pathlib.Path(str(result['actual_output_file_loc']))
    assert actual.stem == 'out_000001', (
        f'_create_video must report the relocated output; got {actual}'
    )
    assert actual.exists() and actual.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tiling inference reads the Tile column; tile-shaped user text in NAMES
# cannot fake a tiling (the old parse-based false positive).
# ---------------------------------------------------------------------------


def test_loader_infers_tiling_from_tile_column(tmp_path):
    rows = ''
    rows += _step_row('A1_BF_TA1', 'A1', 'A1', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('A1_BF_TA2', 'A1', 'A2', -1, 0, 47.9, 34.6, 4972.9)
    rows += _step_row('A1_BF_TB1', 'A1', 'B1', -1, 0, 46.5, 35.9, 4972.9)
    rows += _step_row('A1_BF_TB2', 'A1', 'B2', -1, 0, 47.9, 35.9, 4972.9)
    tsv = tmp_path / 'tiled.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto._config['tiling'] == '2x2'


def test_tile_shaped_names_with_empty_tile_column_infer_no_tiling(tmp_path):
    from modules.tiling_config import TilingConfig

    # User step names embed tile-shaped segments, but the authoritative
    # Tile column is empty: no tiling may be inferred. The old name-parse
    # inference reported 2x2 here and the UI then refused to apply tiling
    # to an "already-tiled" protocol.
    rows = ''
    rows += _step_row('Region_TA1', 'A1', '', -1, -1, 46.5, 34.6, 4972.9)
    rows += _step_row('Region_TA2', 'A2', '', -1, -1, 47.9, 34.6, 4972.9)
    rows += _step_row('Region_TB1', 'A3', '', -1, -1, 46.5, 35.9, 4972.9)
    rows += _step_row('Region_TB2', 'A4', '', -1, -1, 47.9, 35.9, 4972.9)
    tsv = tmp_path / 'tile_shaped_names.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    tc = TilingConfig(tiling_configs_file_loc=TILING_CONFIGS)
    # Untiled inference is falsy-or-1x1; consumers apply `inferred or
    # no_tiling_label()` (the same contract the old name-based inference
    # had). Anything else here means a tiling was faked from the names.
    assert (proto._config['tiling'] or tc.no_tiling_label()) == tc.no_tiling_label()
    # The user text itself survives as the labels.
    assert list(proto.steps()['Label']) == ['Region_TA1', 'Region_TA2', 'Region_TB1', 'Region_TB2']


def test_tiling_inference_skips_malformed_tile_cells_with_warning(monkeypatch):
    from modules import tiling_config as tc_mod
    from modules.tiling_config import TilingConfig

    warnings: list = []

    class _RecordingLogger:
        def warning(self, msg, *args, **kwargs):
            warnings.append(msg % args if args else msg)

    monkeypatch.setattr(tc_mod, 'logger', _RecordingLogger())
    tc = TilingConfig(tiling_configs_file_loc=TILING_CONFIGS)

    # Hand-edited / corrupt cells must be skipped with a warning, never
    # raise out of the load path.
    assert tc.determine_tiling_label_from_tiles(['AA', 'A1B', 'A 1']) is None
    assert warnings and 'malformed' in warnings[0].lower(), warnings

    # Valid cells still infer, with malformed ones contributing nothing.
    warnings.clear()
    assert tc.determine_tiling_label_from_tiles(['A1', 'A2', 'B1', 'B2', 'A 1']) == '2x2'
    assert warnings and 'malformed' in warnings[0].lower(), warnings


# ---------------------------------------------------------------------------
# Label re-sanitization at the load boundary is loud, and the collision
# checks see the SANITIZED labels (two labels differing only in stripped
# characters collide on disk).
# ---------------------------------------------------------------------------

_V8_HEADER = (
    'LumaViewPro Protocol\n'
    'Version\t8\n'
    'Period\t0\n'
    'Duration\t0\n'
    'Labware\tCenter Plate\n'
    'Capture Root\t\n'
    'Steps\n'
    'Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\tAuto_Gain\t'
    'Exposure\tSum\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\t'
    'Z-Stack Group ID\tAcquire\tVideo Config\tStim_Config\tAuto_Named\tLabel\n'
)


def _v8_row(name, label, well, tile_group=0, x=1.0):
    return (
        f'{name}\t{x}\t1.0\t100.0\tFalse\tBF\tFalse\t100.0\t0.0\tFalse\t50.0\t1\t4x\t'
        f'{well}\t\t-1\tFalse\t{tile_group}\t-1\timage\t{{"fps": 5, "duration": 5}}\t{{}}\t'
        f'False\t{label}\n'
    )


def _capture_protocol_logger(monkeypatch):
    from modules import protocol as protocol_mod

    warnings: list = []

    class _RecordingLogger:
        def warning(self, msg, *args, **kwargs):
            warnings.append(msg % args if args else msg)

        def info(self, msg, *args, **kwargs):
            pass

        def error(self, msg, *args, **kwargs):
            pass

    monkeypatch.setattr(protocol_mod, 'logger', _RecordingLogger())
    return warnings


def test_load_sanitizes_labels_loudly(tmp_path, monkeypatch):
    warnings = _capture_protocol_logger(monkeypatch)
    tsv = tmp_path / 'dotted_label.tsv'
    tsv.write_text(_V8_HEADER + _v8_row('A.1_BF', 'A.1', 'B2'))

    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    step = proto.step(idx=0)
    assert step['Label'] == 'A1', 'the stripped-character form is what the writer saves'
    assert step['Name'] == 'A1_BF'
    assert any('removed unsupported characters' in w for w in warnings), warnings


def test_labels_differing_only_in_stripped_chars_collide(tmp_path, monkeypatch):
    # 'A.1' and 'A1' sanitize to one label; on different wells they render
    # one capture base, so the load warns and the run is refused at start.
    from modules import protocol as protocol_mod

    notified: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            notified.append(message)

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    tsv = tmp_path / 'stripped_collision.tsv'
    tsv.write_text(
        _V8_HEADER
        + _v8_row('A.1_BF', 'A.1', 'B2', tile_group=0, x=1.0)
        + _v8_row('A1_BF', 'A1', 'B3', tile_group=1, x=2.0)
    )
    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert list(proto.steps()['Label']) == ['A1', 'A1']
    assert len(notified) == 1 and 'refused' in notified[0].lower(), notified

    errors = proto.validate_for_run(axis_limits=None)
    assert any('would save captures' in e for e in errors), errors


# ---------------------------------------------------------------------------
# Objective disambiguation: the collision key pairs the rendered base with
# the Objective column (turret builds stamp the objective per step).
# ---------------------------------------------------------------------------


def test_same_base_different_objective_is_not_a_collision(tmp_path, monkeypatch):
    from modules import protocol as protocol_mod

    notified: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            notified.append(message)

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    # Load leg: same rendered base, different Objective -> no warning.
    rows = ''
    rows += _step_row('A1_BF', 'A1', '', -1, 0, 46.5, 34.6, 4972.9).replace('20x Oly', '4x Oly')
    rows += _step_row('A1_BF', 'A1', '', -1, 1, 46.5, 34.6, 5001.7)
    tsv = tmp_path / 'objective_disambiguated.tsv'
    tsv.write_text(_build_tsv(rows))
    proto = Protocol.from_file(file_path=tsv, tiling_configs_file_loc=TILING_CONFIGS)
    assert proto.num_steps() == 2
    assert notified == [], notified

    # Run leg: no collision errors either.
    p = _make_protocol(
        [
            _valid_step(Well='A1', X=60.0, Y=40.0, Z=5000.0, Objective='4x Oly'),
            _valid_step(Well='A1', X=60.0, Y=40.0, Z=5000.0, Objective='20x Oly'),
        ]
    )
    errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS, stage_offset=_STAGE_OFFSET)
    assert not any('would save captures' in e for e in errors), errors


def test_same_base_same_objective_still_refused_at_run_start():
    p = _make_protocol(
        [
            _valid_step(Well='A1', X=60.0, Y=40.0, Z=5000.0, Objective='4x Oly'),
            _valid_step(Well='A1', X=60.0, Y=40.0, Z=5000.0, Objective='4x Oly'),
        ]
    )
    errors = p.validate_for_run(axis_limits=_DEFAULT_AXIS_LIMITS, stage_offset=_STAGE_OFFSET)
    collision_errors = [e for e in errors if 'would save captures' in e]
    assert len(collision_errors) == 1, errors
    assert 'Steps 1, 2' in collision_errors[0]


# ---------------------------------------------------------------------------
# UI routing: a 'collision' refusal surfaces its own message, never the
# generic "No Z-Stack data found" folder advice.
# ---------------------------------------------------------------------------


def test_zprojection_callback_routes_collision_to_failure_message():
    import ast

    src_path = REPO / 'ui' / 'post_processing.py'
    tree = ast.parse(src_path.read_text())
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ZProjectionControls':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'zprojection_callback':
                    method = child
    assert method is not None, 'ZProjectionControls.zprojection_callback not found'
    src = ast.unparse(method)
    assert "('error', 'collision')" in src, (
        "a reason='collision' result must take the failed-with-message branch "
        '(its message carries the real remedy), not the pick-a-different-folder path'
    )
