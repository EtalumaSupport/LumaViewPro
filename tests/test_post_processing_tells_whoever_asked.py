# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build tells whoever asked what it made, once.

Before, a build answered with a dict whose failure half was untyped prose
(`status: False`), a build where some groups failed answered 'Success.', a
video's missing frames were told only by a popup, and the loader posted its
own notices whenever it guessed nobody was watching -- so the stitcher
plugin, which also returns its result, reported twice, and the GUI's
z-projection told one failure twice.

Now `load_folder` returns only a complete result and raises everything
else: a refusal when the folder cannot yield the output, a failure naming
what is missing (and carrying what was produced) otherwise. It posts
nothing. The one build nobody waits on, the post-run hyperstack, announces
itself and answers under the same operation key. The GUI's buttons go
through the GUI boundary, so a failure is told once per press.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

import modules.stack_builder as stack_builder
from modules.common_utils import PostFunction
from modules.exceptions import (
    CaptureError,
    PostProcessingFailedError,
    PostProcessingRefusedError,
)
from modules.notification_center import NotificationCenter, Severity, notifications
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from tests.test_capture_collision_policy import TILING_CONFIGS


class _Processor(ProtocolPostProcessor):
    """Drives the base class's load_folder with scripted group outcomes."""

    def __init__(self, outcomes, post_function=PostFunction.HYPERSTACK):
        super().__init__(post_function=post_function, has_turret=False)
        self._outcomes = list(outcomes)

    def _get_groups(self, df):
        return [(key, group) for key, group in df.groupby('GroupKey')]

    def _generate_filename(self, df, **kwargs):
        return df.iloc[0]['OutName']

    def _filter_ignored_types(self, df):
        return df

    def _group_algorithm(self, path, df, **kwargs):
        return self._outcomes.pop(0)

    def _add_record(self, protocol_post_record, alg_metadata, root_path, **kwargs):
        pass


def _images_df(groups):
    rows = []
    for g in range(groups):
        for i in range(2):  # two frames per group, so each group is eligible
            row = {'Filepath': f'g{g}_f{i}.tiff', 'GroupKey': g, 'OutName': f'out{g}.tiff'}
            row.update(dict.fromkeys(PostFunction.list_values(), False))
            rows.append(row)
    return pd.DataFrame(rows)


def _drive(processor, tmp_path, monkeypatch, *, helper=None, groups=2):
    post_record = MagicMock()
    post_record.file_exists_in_records.return_value = False
    answer = helper or {
        'status': True,
        'images_df': _images_df(groups),
        'root_path': tmp_path,
        'protocol_post_record': post_record,
        'protocol': None,
    }
    monkeypatch.setattr(processor._post_processing_helper, 'load_folder', lambda **kw: answer)
    return processor.load_folder(path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS)


@pytest.fixture
def posts(monkeypatch):
    """Every notification posted, of any severity, as (method, title, kwargs)."""
    seen = []
    for method in ('notice', 'info', 'warning', 'error', 'critical'):
        monkeypatch.setattr(
            notifications,
            method,
            lambda category, title, message, _m=method, **kw: seen.append((_m, title, kw)),
        )
    return seen


def _ok(**metadata):
    return PostProcResult.ok(significant_bits=8, record_metadata=metadata or None)


# ---------------------------------------------------------------------------
# load_folder: complete, or it raises
# ---------------------------------------------------------------------------


def test_a_build_where_some_groups_failed_raises_with_what_it_made(tmp_path, monkeypatch, posts):
    processor = _Processor(
        [_ok(), PostProcResult.failed('bad tile'), PostProcResult.failed('bad plane')]
    )

    with pytest.raises(PostProcessingFailedError) as raised:
        _drive(processor, tmp_path, monkeypatch, groups=3)

    fault = raised.value
    assert fault.reason == 'post_processing_incomplete'
    assert fault.title == 'Hyperstack Incomplete'
    assert fault.produced_paths == (str(tmp_path / 'Hyperstack' / 'out0.tiff'),)
    assert [e.split(': ', 1)[1] for e in fault.errors] == ['bad tile', 'bad plane']
    assert '2 of 3 hyperstack group(s) failed' in str(fault)
    assert posts == []


def test_a_build_where_every_group_failed_names_every_error(tmp_path, monkeypatch, posts):
    processor = _Processor([PostProcResult.failed('one'), PostProcResult.failed('two')])

    with pytest.raises(PostProcessingFailedError) as raised:
        _drive(processor, tmp_path, monkeypatch)

    fault = raised.value
    assert fault.reason == 'post_processing_failed'
    assert fault.title == 'Hyperstack Failed'
    assert fault.produced_paths == ()
    assert len(fault.errors) == 2
    assert 'Nothing was saved.' in str(fault)


def test_a_video_short_of_frames_raises_with_the_video(tmp_path, monkeypatch, posts):
    processor = _Processor([_ok(dropped_frames=3), _ok()], post_function=PostFunction.VIDEO)

    with pytest.raises(PostProcessingFailedError) as raised:
        _drive(processor, tmp_path, monkeypatch)

    fault = raised.value
    assert len(fault.produced_paths) == 2
    assert '3 frame(s) could not be added' in str(fault)
    assert posts == []


def test_a_complete_build_returns_and_posts_nothing(tmp_path, monkeypatch, posts):
    result = _drive(_Processor([_ok(), _ok()]), tmp_path, monkeypatch)

    assert result['status'] is True
    assert result['new_count'] == 2
    assert posts == []


def test_a_folder_with_no_images_is_refused(tmp_path, monkeypatch, posts):
    helper = {
        'status': True,
        'images_df': _images_df(0),
        'root_path': tmp_path,
        'protocol_post_record': MagicMock(),
        'protocol': None,
    }

    with pytest.raises(PostProcessingRefusedError) as raised:
        _drive(_Processor([]), tmp_path, monkeypatch, helper=helper)

    assert raised.value.reason == 'no_images'
    assert raised.value.title == 'Hyperstack Not Possible'
    assert posts == []


def test_the_helpers_own_sentence_reaches_the_refusal(tmp_path, monkeypatch, posts):
    helper = {'status': False, 'message': 'Protocol and/or Protocol Record not found in folder'}

    with pytest.raises(PostProcessingRefusedError) as raised:
        _drive(_Processor([]), tmp_path, monkeypatch, helper=helper)

    assert raised.value.reason == 'protocol_data_unreadable'
    assert str(raised.value) == 'Protocol and/or Protocol Record not found in folder'


def test_a_z_projection_with_no_z_stack_says_where_one_lives():
    refusal = PostProcessingRefusedError(
        operation='Z-Projection', reason='no_data', message='No ZProject was generated.'
    )

    assert refusal.title == 'No Z-Stack Data Found'
    assert 'Pick a folder that contains a Z-stack run' in str(refusal)


# ---------------------------------------------------------------------------
# The post-run hyperstack: the one build nobody waits on
# ---------------------------------------------------------------------------


def _hyperstack_build(monkeypatch, tmp_path, answer):
    reports = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exception, **kw: reports.append((exception, kw))
    )

    def _load_folder(self, **kwargs):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(stack_builder.StackBuilder, 'load_folder', _load_folder)
    # The run's images are already on disk: the wait returns at once.
    stack_builder.build_hyperstacks_for_run(
        tmp_path, False, TILING_CONFIGS, wait_for_images=lambda: None
    )
    return reports


def test_the_post_run_build_announces_then_answers_under_one_key(tmp_path, monkeypatch, posts):
    answer = {
        'status': True,
        'message': 'Success.',
        'new_count': 4,
        'output_root': str(tmp_path),
        'artifact_paths': [],
        'accounting_note': '',
    }

    reports = _hyperstack_build(monkeypatch, tmp_path, answer)

    assert [(m, t) for m, t, _ in posts] == [
        ('notice', 'Saving Hyperstacks'),
        ('notice', 'Hyperstacks Saved'),
    ]
    assert {kw['operation_key'] for _, _, kw in posts} == {'post-processing:Hyperstack'}
    assert reports == []


def test_the_post_run_builds_failure_replaces_its_announcement_in_its_own_words(
    tmp_path, monkeypatch, posts
):
    fault = CaptureError('Plane 3 of well A1 could not be read.', 'unreadable_input_frame')

    reports = _hyperstack_build(monkeypatch, tmp_path, fault)

    assert [(m, t) for m, t, _ in posts] == [('notice', 'Saving Hyperstacks')]
    ((reported, kw),) = reports
    assert reported is fault
    assert kw['solicited'] is False
    assert kw['operation_key'] == 'post-processing:Hyperstack'


def test_report_outcome_hands_a_faults_operation_key_to_the_popup():
    center = NotificationCenter()
    shown = []
    center.add_listener(shown.append, min_severity=Severity.WARNING)

    center.report_outcome(
        PostProcessingFailedError(operation='Stitch', missing='2 of 9 stitch group(s) failed.'),
        solicited=False,
        category='Post-processing',
        operation_key='post-processing:Stitched',
    )

    ((notification,),) = [shown]
    assert notification.operation_key == 'post-processing:Stitched'
    assert notification.title == 'Stitch Failed'


# ---------------------------------------------------------------------------
# The callers that wait
# ---------------------------------------------------------------------------


def test_a_gui_build_that_raises_is_told_once_as_the_persons_request(monkeypatch):
    import ui.post_processing as post_processing
    import ui.ui_helpers as ui_helpers

    reports = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exception, **kw: reports.append((exception, kw))
    )

    def _run_inline(call, redraw, label, **kw):
        ui_helpers._reported(call, label)
        redraw()

    monkeypatch.setattr(post_processing, 'submit_reported', _run_inline)
    monkeypatch.setattr(post_processing._app_ctx, 'ctx', MagicMock())
    refusal = PostProcessingRefusedError(operation='Stitch', reason='no_data', message='Nothing.')
    popup = MagicMock()
    shown = []

    def _build(progress):
        raise refusal

    post_processing._run_build(_build, popup, 'RUN_STITCHER', on_done=shown.append)

    assert reports == [(refusal, {'solicited': True, 'category': 'UI:RUN_STITCHER'})]
    assert shown == [None]
    popup.dismiss.assert_called_once_with()


def test_the_stitcher_plugin_answers_a_typed_failure_in_its_words_with_its_outputs(
    tmp_path,
):
    from modules.plugins.builtin import stitcher_plugin

    fault = PostProcessingFailedError(
        operation='Stitch',
        missing='1 of 2 stitch group(s) failed (A1: bad tile).',
        produced_paths=[str(tmp_path / 'B1.tiff')],
        output_root=str(tmp_path),
    )

    post_processing = MagicMock()
    post_processing.stitch.side_effect = fault

    result = stitcher_plugin._stitcher_processor(post_processing, str(tmp_path), {}, '')

    assert result.success is False
    assert result.message == str(fault)
    assert result.outputs == (str(tmp_path / 'B1.tiff'),)


def test_a_composite_run_carries_the_builders_own_reason(tmp_path, monkeypatch):
    import modules.composite_generation as composite_generation
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    def _refuse(self, **kwargs):
        raise PostProcessingRefusedError(
            operation='Composite', reason='excluded_inputs', message='Only composites here.'
        )

    monkeypatch.setattr(composite_generation.CompositeGeneration, 'load_folder', _refuse)

    with (
        open_composite_session(headless_settings(tmp_path)) as (_session, runner),
        pytest.raises(CaptureError) as raised,
    ):
        runner.run_composite(sequence_name='refused', parent_dir=str(tmp_path))

    assert raised.value.reason == 'excluded_inputs'


# ---------------------------------------------------------------------------
# The manual recording's video: the same contract as a protocol folder
# ---------------------------------------------------------------------------


def _manual_video(monkeypatch, tmp_path, video_result):
    import modules.video_builder as video_builder

    builder = video_builder.VideoBuilder(has_turret=False)
    monkeypatch.setattr(
        video_builder.image_utils, 'find_tiff_files', lambda path: [tmp_path / 'frame.tiff']
    )
    monkeypatch.setattr(video_builder.recording_frames, 'is_manual_video_frame', lambda n: True)
    monkeypatch.setattr(video_builder.recording_frames, 'frame_number', lambda n: 0)
    monkeypatch.setattr(builder, '_read_recording_manifest', lambda path: {'channel_color': None})
    monkeypatch.setattr(builder, '_create_video', lambda **kwargs: video_result)
    return builder._build_manual_recording_video(tmp_path, frames_per_sec=10)


def test_a_manual_video_that_failed_to_encode_raises(tmp_path, monkeypatch, posts):
    with pytest.raises(PostProcessingFailedError) as raised:
        _manual_video(monkeypatch, tmp_path, {'status': False, 'error': 'encoder died'})

    assert 'encoder died' in str(raised.value)
    assert raised.value.produced_paths == ()


def test_a_manual_video_short_of_frames_raises_with_the_video(tmp_path, monkeypatch, posts):
    video = tmp_path / 'rec.mp4'

    with pytest.raises(PostProcessingFailedError) as raised:
        _manual_video(
            monkeypatch,
            tmp_path,
            {
                'status': True,
                'actual_output_file_loc': video,
                'metadata': {'dropped_frames': 2},
            },
        )

    assert raised.value.produced_paths == (str(video),)
    assert '2 frame(s) could not be added' in str(raised.value)
