# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Create Video's 'auto' rate plays a recording as it was recorded.

'auto' (frame_rate_fps=None) is decided for each video from its own frames.
A manual recording whose first frame states its time plays each frame at
the time it was taken, so a lost frame shows as a gap; a later frame with
no time after the one before it is left out and counted. One whose first
frame states none plays evenly at its manifest's measured rate -- engine
manifests first, the legacy session manifest for pre-engine folders --
and a folder with no measured rate builds at the named default. An
explicit rate still overrides everything (deliberate timelapse speedups
stay possible). A protocol run's video step saved as frames plays by the
same rule; the run's stills play at the default.
"""

import datetime
import json
import logging
import pathlib
from collections import Counter
from unittest.mock import patch

import av
import numpy as np
import pytest
import tifffile as tf

import _lumascope.modules.image_utils as image_utils
from _lumascope.modules.exceptions import PostProcessingFailedError
from _lumascope.modules.video_builder import DEFAULT_BUILD_FPS, VideoBuilder
from _lumascope.modules.channel_record import release_colors
from tests.log_capture import records_of
from tests.stock_channels import LED0, LED1, LED2
from tests.installation_fixtures import shipped_grids
from tests.frame_records import recorded_record, write_recorded
from tests.recorded_run_folders import Recording, Still, run_folder
from _lumascope.modules.lumascope_api.frame_record import FrameTime, TimestampSource


def _make_manual_folder(tmp_path, n=3):
    for i in range(n):
        (tmp_path / f'ManualVideo_Frame_{i:04}_2026-08-08_12-00-00-000.tiff').touch()
    return tmp_path


def _built_rate(tmp_path, frame_rate_fps):
    captured = {}

    def _capture(self, path, df, frame_rate_fps, **kwargs):
        captured['fps'] = frame_rate_fps
        return {'status': True, 'actual_output_file_loc': path / f'{path.name}.mp4'}

    with patch.object(VideoBuilder, '_create_video', _capture):
        builder = VideoBuilder(has_turret=False)
        result = builder.build_from_folder(
            tmp_path,
            shipped_grids(),
            frame_rate_fps=frame_rate_fps,
            enable_timestamp_overlay=False,
            channel_colors=release_colors(),
        )
    assert result['status'] is True
    return captured['fps']


def test_auto_uses_engine_manifest_measured_rate(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(
        json.dumps({'measured_fps': 9.5, 'channel_color': LED0})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(9.5)


def test_auto_uses_legacy_manifest_measured_rate(tmp_path):
    # A pre-engine folder's manifest, as an earlier release wrote it: old channel names.
    _make_manual_folder(tmp_path)
    (tmp_path / 'session_manifest.json').write_text(
        json.dumps({'recording': {'channel_color': 'Red', 'actual_fps': {'mean': 3.25}}})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(3.25)


def test_auto_without_any_manifest_uses_named_default(tmp_path):
    _make_manual_folder(tmp_path)
    assert _built_rate(tmp_path, None) == DEFAULT_BUILD_FPS


def test_explicit_rate_overrides_the_manifest(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 9.5}))
    assert _built_rate(tmp_path, 30) == 30


def test_engine_manifest_wins_over_legacy(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 9.5}))
    (tmp_path / 'session_manifest.json').write_text(
        json.dumps({'recording': {'actual_fps': {'mean': 3.25}}})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(9.5)


def test_a_manifest_names_its_channel_and_whether_it_was_false_coloured(tmp_path):
    """A recording records its channel and the false-colour flag; the video
    is coloured by the channel only when the flag was on."""
    builder = VideoBuilder(has_turret=False)
    read = []
    for false_color in (True, False):
        folder = tmp_path / str(false_color)
        folder.mkdir()
        (folder / 'recording_manifest.json').write_text(
            json.dumps({'channel': LED2, 'false_color': false_color, 'measured_fps': 2.0})
        )
        read.append(builder._read_recording_manifest(folder)['channel_color'])
    assert read == [LED2, None]


def test_channel_color_read_from_either_manifest_generation(tmp_path):
    builder = VideoBuilder(has_turret=False)
    (tmp_path / 'recording_manifest.json').write_text(
        json.dumps({'channel_color': LED1, 'measured_fps': 2.0})
    )
    info = builder._read_recording_manifest(tmp_path)
    assert info == {'channel_color': LED1, 'measured_fps': 2.0}

    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    (legacy_dir / 'session_manifest.json').write_text(
        json.dumps({'recording': {'channel_color': 'Lumi', 'actual_fps': {'mean': 0.0}}})
    )
    info = builder._read_recording_manifest(legacy_dir)
    # A zero-sample mean is not a measured rate.
    assert info == {'channel_color': 'Lumi', 'measured_fps': None}


def test_frames_without_any_manifest_warn_loudly(tmp_path, caplog):
    # The manifest is the sole carrier of the recording's channel color
    # and measured rate. Silently returning None fields made a LOST
    # manifest indistinguishable from deliberate grayscale: the build
    # quietly produced a colorless video at the default rate. The reader
    # now says so in the log (log-only: a pre-manifest legacy folder
    # takes the same path legitimately, so a popup would misfire).
    import _lumascope.modules.video_builder as video_builder_module

    builder = VideoBuilder(has_turret=False)
    with caplog.at_level(logging.WARNING):
        info = builder._read_recording_manifest(tmp_path)

    assert info == {'channel_color': None, 'measured_fps': None}
    warned = [
        r.getMessage()
        for r in records_of(caplog, video_builder_module)
        if r.levelno == logging.WARNING
    ]
    assert warned, 'a folder with no readable manifest must warn about the degraded build'
    message = warned[-1]
    assert 'manifest' in message.lower() and 'grayscale' in message.lower()


def _write_timed_frames(scope, folder, seconds, first=0):
    """Recorded frames through the one recorded-frame writer, numbered from
    ``first``, each at its entry of ``seconds``."""
    base = datetime.datetime(2026, 11, 12, 16, 35, 0).astimezone()
    for n, s in enumerate(seconds, start=first):
        at = base + datetime.timedelta(seconds=s)
        stamp = at.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        write_recorded(
            scope,
            np.full((64, 64), 40 + n, dtype=np.uint8),
            folder / f'ManualVideo_Frame_{n:04}_{stamp}.tiff',
            record=recorded_record(
                captured_at=FrameTime(at=at, source=TimestampSource.HOST_ARRIVAL)
            ),
            frame_number=n,
        )


def _frame_times_s(video_path) -> list[float]:
    with av.open(str(video_path)) as container:
        return [float(frame.time) for frame in container.decode(video=0)]


def _build(folder, frame_rate_fps=None):
    return VideoBuilder(has_turret=False).build_from_folder(
        folder,
        shipped_grids(),
        frame_rate_fps=frame_rate_fps,
        enable_timestamp_overlay=False,
        channel_colors=release_colors(),
    )


def test_auto_plays_each_frame_at_its_own_time_so_a_lost_frame_is_a_gap(sim_scope, tmp_path):
    # Frames 0-2 a tenth of a second apart, then a second lost, then two more.
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.1, 0.2, 1.2, 1.3])
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 10.0}))

    result = _build(tmp_path)

    times = _frame_times_s(result['artifact_paths'][0])
    assert times == pytest.approx([0.0, 0.1, 0.2, 1.2, 1.3], abs=1e-3)


def test_a_chosen_rate_plays_the_frames_evenly_at_it(sim_scope, tmp_path):
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.1, 0.2, 1.2, 1.3])

    result = _build(tmp_path, frame_rate_fps=10)

    times = _frame_times_s(result['artifact_paths'][0])
    assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4], abs=1e-3)


def _write_untimed_frame(folder, n, stamp):
    """A frame another producer wrote: pixels and no record."""
    tf.imwrite(
        str(folder / f'ManualVideo_Frame_{n:04}_{stamp}.tiff'),
        np.full((64, 64), 9, dtype=np.uint8),
    )


def test_a_first_frame_with_no_time_builds_the_folder_evenly_at_the_measured_rate(
    sim_scope, tmp_path
):
    _write_untimed_frame(tmp_path, 0, '2026-11-12_16-34-59-000')
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.1, 0.2, 1.2], first=1)
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 4.0}))

    result = _build(tmp_path)

    times = _frame_times_s(result['artifact_paths'][0])
    assert times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-3)


def test_a_later_frame_with_no_time_is_left_out_and_counted(sim_scope, tmp_path):
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.1, 0.2, 1.2])
    _write_untimed_frame(tmp_path, 4, '2026-11-12_16-35-01-300')
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 4.0}))

    with pytest.raises(PostProcessingFailedError) as incomplete:
        _build(tmp_path)

    assert '1 frame(s) could not be added' in str(incomplete.value)
    video = incomplete.value.produced_paths[0]
    assert _frame_times_s(video) == pytest.approx([0.0, 0.1, 0.2, 1.2], abs=1e-3)


def test_a_frame_at_or_before_the_one_before_it_is_left_out_and_counted(sim_scope, tmp_path):
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.2, 0.1, 0.2, 0.3])
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 5.0}))

    with pytest.raises(PostProcessingFailedError) as incomplete:
        _build(tmp_path)

    assert '2 frame(s) could not be added' in str(incomplete.value)
    video = incomplete.value.produced_paths[0]
    assert _frame_times_s(video) == pytest.approx([0.0, 0.2, 0.3], abs=1e-3)


def test_auto_rate_opens_each_frame_once_but_the_first(sim_scope, tmp_path, monkeypatch):
    # Reading every frame's time in a pass before the encode opened each file
    # twice, a disk seek a frame on a slow PC (176 s of 840 on 9,105 frames).
    # The first frame is read once more, to fix the writer's timing.
    _write_timed_frames(sim_scope, tmp_path, [0.0, 0.1, 0.2, 1.2, 1.3])
    opens = Counter()
    real_tifffile = image_utils.tf.TiffFile

    def counting_tifffile(path, *args, **kwargs):
        opens[pathlib.Path(path).name] += 1
        return real_tifffile(path, *args, **kwargs)

    monkeypatch.setattr(image_utils.tf, 'TiffFile', counting_tifffile)

    _build(tmp_path)

    names = sorted(opens)
    assert len(names) == 5
    assert [opens[name] for name in names] == [2, 1, 1, 1, 1], dict(opens)


# A protocol run's video steps saved as frames play by the same rule as a
# manual recording; its stills, recorded at no playback rate, at the default.


def _protocol_build(folder, frame_rate_fps=None):
    result = _build(folder, frame_rate_fps)
    return sorted(pathlib.Path(p) for p in result['artifact_paths'])


def test_a_protocol_recording_plays_each_frame_at_its_own_time_so_a_lost_frame_is_a_gap(
    sim_scope, tmp_path
):
    # 40 fps, the fourth frame lost.
    seconds = (0.0, 0.025, 0.05, 0.1, 0.125)
    folder = run_folder(sim_scope, tmp_path, [Recording('A1_BF', seconds, measured_fps=39.5)])

    (video,) = _protocol_build(folder)

    assert _frame_times_s(video) == pytest.approx(list(seconds), abs=1e-3)


def test_a_protocol_recording_with_no_frame_times_plays_evenly_at_its_measured_rate(
    sim_scope, tmp_path
):
    folder = run_folder(
        sim_scope,
        tmp_path,
        [Recording('A1_BF', (0.0, 0.1, 0.2, 0.3), timed=False, measured_fps=40.0)],
    )

    (video,) = _protocol_build(folder)

    assert _frame_times_s(video) == pytest.approx([0.0, 0.025, 0.05, 0.075], abs=1e-3)


def test_a_protocol_timelapse_of_stills_plays_at_the_default_rate_under_its_name(
    sim_scope, tmp_path
):
    folder = run_folder(sim_scope, tmp_path, [Still('A1_BF', scan) for scan in range(3)])

    (video,) = _protocol_build(folder)

    step = 1 / DEFAULT_BUILD_FPS
    assert _frame_times_s(video) == pytest.approx([0.0, step, 2 * step], abs=1e-3)
    # The name a build of the release before gave it (its settings' digest
    # included), so a rebuild finds it made.
    assert video.name == 'A1_BF_Z0_video_08b868.mp4'


def test_a_chosen_rate_plays_a_protocol_recording_evenly_at_it(sim_scope, tmp_path):
    folder = run_folder(
        sim_scope,
        tmp_path,
        [Recording('A1_BF', (0.0, 0.025, 0.05, 0.1), measured_fps=39.5)],
    )

    (video,) = _protocol_build(folder, frame_rate_fps=10)

    assert _frame_times_s(video) == pytest.approx([0.0, 0.1, 0.2, 0.3], abs=1e-3)


def test_two_video_steps_at_one_place_in_one_scan_build_a_video_each(sim_scope, tmp_path):
    # Two added steps at one well, position, Z-slice and channel, told apart
    # only by their labels and their recordings' folders.
    first = (0.0, 0.1, 0.2)
    second = (0.0, 0.5, 1.0, 1.5)
    folder = run_folder(
        sim_scope,
        tmp_path,
        [
            Recording('custom0001_BF', first, label='custom0001', custom=True, measured_fps=10.0),
            Recording('custom0002_BF', second, label='custom0002', custom=True, measured_fps=2.0),
        ],
    )

    videos = _protocol_build(folder)

    assert [_frame_times_s(v) for v in videos] == [
        pytest.approx(list(first), abs=1e-3),
        pytest.approx(list(second), abs=1e-3),
    ]
