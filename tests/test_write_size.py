# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""What a run writes is sized from the frames it saves, and a run too big for
its disk is refused at its start (#843, #869, audit M6).

Every image and frame was sized at a flat 8 MB whatever its size and depth
(#869: one scan of 96 x 30 s at 61 fps, 1400x1400 8-bit, as frames, sized
4.3x high), and only one scan was ever sized, while a run's size is every
scan of it (a run that ran 7 h and wrote 56 GB before the per-scan check
stopped it; the advisory 144x low on 48 h / 20 min). The owner is
``_lumascope.modules.write_size``; ``prepare()`` refuses a run whose whole
size does not fit.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from _lumascope.modules import channel_record, image_mode, image_utils
from _lumascope.modules.exceptions import ProtocolRunRefusedError
from _lumascope.modules.lumascope_api.frame_record import CapturedStill
from _lumascope.modules.scope_capabilities import FrameSize
from _lumascope.modules.sequenced_capture_runner import SequencedCaptureRunMode
from _lumascope.modules.write_size import FrameSizing, estimate_step_write_mb
from _lumascope.modules.image_mode import RecordingFormat
from tests.disk_fakes import free_space_unreadable
from tests.frame_records import frame_record, unpositioned
from tests.protocol_drives import a_frame_sizing
from tests.scope_fakes import bind_settings_like_a_session, build_scope
from tests.test_audit_fixes import _bare_protocol_writer, _protocol_step
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    executor,
    executors,
    scope,
)
from tests.test_protocol_roundtrip import _build_protocol, _make_step

_MIB = 1024**2


# --- the bytes a saved frame holds ------------------------------------------


def _saved_by_the_writers(scope, mode, sum_count, channel, false_color_on):
    """The pixels a run's still writer hands the TIFF, as it builds them: the
    capture at the mode's depth, then the save encoding's false colour."""
    config = image_mode.ImageCaptureConfig.from_image_mode(mode)
    still = scope.imaging.capture_still(
        force_to_8bit=config.capture_depth == 8,
        sum_count=sum_count,
        accept_dark=True,
        timeout_s=2.0,
    )
    color = channel_record.shown_color(channel, false_color_on, channel_record.release_colors())
    return image_utils.maybe_apply_false_color(
        data=still.image,
        color=color,
        use_false_color_16bit=config.save_encoding == image_mode.SAVE_ENCODING_RGB,
    )


@pytest.mark.slow
def test_a_saved_frame_is_sized_at_the_bytes_the_writers_save():
    """On an 8-bit camera (an LS620's FX2) and a 12-bit one, in every image mode, a single frame
    and a sum, a channel shown in colour with false colour on and off and a
    grey one: the rule's bytes equal the array the writers save."""
    offenders = []
    for model in ('LS620', 'LS850T'):
        scope = build_scope(simulate=True, sim_model=model)
        bind_settings_like_a_session(scope)
        scope.imaging.enable_streaming()
        camera_bits = scope.capabilities.camera_max_significant_bits
        try:
            for mode in image_mode.ImageMode:
                scope.camera.set_image_mode(mode)
                config = image_mode.ImageCaptureConfig.from_image_mode(mode)
                for sum_count in (1, 3):
                    for channel, false_color_on in (
                        ('Ex488', True),
                        ('Ex488', False),
                        ('BF', True),
                    ):
                        saved = _saved_by_the_writers(
                            scope, mode, sum_count, channel, false_color_on
                        )
                        sizing = a_frame_sizing(
                            frame=FrameSize(saved.shape[1], saved.shape[0]),
                            camera_bits=camera_bits,
                            capture_depth=config.capture_depth,
                            save_encoding=config.save_encoding,
                        )
                        sized = sizing.saved_frame_bytes(
                            channel=channel, sum_count=sum_count, false_color_on=false_color_on
                        )
                        if sized != saved.nbytes:
                            offenders.append(
                                f'{model} {mode} sum {sum_count} {channel} '
                                f'false colour {false_color_on}: sized {sized}, saved {saved.nbytes}'
                            )
        finally:
            scope.disconnect()
    assert not offenders, '\n'.join(offenders)


@pytest.mark.parametrize(
    ('sim_session', 'camera_bits'),
    [({}, 12), ({'microscope': 'LS620'}, 8)],
    indirect=['sim_session'],
)
# parametrize-justified: each camera is its own session.
def test_a_run_is_sized_for_the_camera_it_saves_from(sim_session, camera_bits):
    """The sizing reads the camera's frame size and the most bits it
    delivers: an FX2 (LS620) saves a full-depth mode's single frame at one
    byte a pixel."""
    scope = sim_session.scope
    sizing = FrameSizing.for_camera(
        scope.camera.frame_size,
        scope.capabilities,
        image_mode.ImageCaptureConfig.from_image_mode('12bit_scientific'),
        sim_session.channel_colors,
    )
    frame = scope.camera.frame_size

    assert sizing.frame == frame
    assert sizing.camera_bits == camera_bits
    assert sizing.saved_frame_bytes(channel='BF', sum_count=1, false_color_on=False) == (
        frame.width * frame.height * (2 if camera_bits > 8 else 1)
    )


# --- one scan, and every scan ------------------------------------------------


def _frames_protocol(n_steps=96, duration_s=30, period_min=1.0, duration_hrs=1.0):
    return _build_protocol(
        [
            _make_step(name=f'V{i}', acquire='video', video_config={'duration': duration_s})
            for i in range(n_steps)
        ],
        period_min=period_min,
        duration_hrs=duration_hrs,
    )


def test_one_scan_of_869s_protocol_is_sized_from_its_frames():
    """#869: 96 steps of 30 s at 61 fps, 1400x1400 8-bit, saved as frames:
    96 x 1830 frames of 1400 x 1400 x 1 byte, not 96 x 1830 x 8 MB."""
    scan_mb = _frames_protocol().estimate_write_mb(
        sizing=a_frame_sizing(), recording_format=RecordingFormat.FRAMES, video_fps=61.0
    )

    assert scan_mb == pytest.approx(96 * 1830 * 1400 * 1400 / _MIB)


def test_each_stack_counts_its_own_files_twice():
    """A stack is built after the run beside the files it reads, which stay:
    the stills' when the Sequenced format stacks, a recording's frames when
    the recording format does, each on its own; an MP4 is not stacked."""
    protocol = _build_protocol(
        [
            _make_step(name='S0'),
            _make_step(name='V0', acquire='video', video_config={'duration': 600}),
        ]
    )

    def scan_mb(stacked, recording_format):
        return protocol.estimate_write_mb(
            sizing=a_frame_sizing(stacked=stacked),
            recording_format=recording_format,
            video_fps=40.0,
        )

    still_mb = estimate_step_write_mb(
        protocol.step_row(step_index=0), sizing=a_frame_sizing(), video_fps=40.0
    )
    frames_mb = estimate_step_write_mb(
        protocol.step_row(step_index=1),
        sizing=a_frame_sizing(),
        recording_format=RecordingFormat.FRAMES,
        video_fps=40.0,
    )
    once = scan_mb(False, RecordingFormat.FRAMES)
    assert once == pytest.approx(still_mb + frames_mb)
    assert scan_mb(True, RecordingFormat.FRAMES) == pytest.approx(once + still_mb)
    assert scan_mb(False, RecordingFormat.HYPERSTACK) == pytest.approx(once + frames_mb)
    assert scan_mb(True, RecordingFormat.HYPERSTACK) == pytest.approx(once + still_mb + frames_mb)
    assert scan_mb(True, RecordingFormat.MP4) == pytest.approx(
        scan_mb(False, RecordingFormat.MP4) + still_mb
    )


def test_an_mp4_step_is_its_frames_at_a_compression_fraction_over_a_floor():
    long_clip = _make_step(
        acquire='video', video_config={'duration': 600}, false_color=True, color='Ex488'
    )
    short_clip = _make_step(acquire='video', video_config={'duration': 1})
    sizing = a_frame_sizing()

    # 600 s at 40 fps of three-plane 8-bit 1400x1400 frames, a tenth of them.
    assert estimate_step_write_mb(long_clip, sizing=sizing, video_fps=40.0) == pytest.approx(
        24_000 * 1400 * 1400 * 3 * 0.1 / _MIB
    )
    assert estimate_step_write_mb(short_clip, sizing=sizing, video_fps=40.0) == 50


def test_a_malformed_video_duration_sizes_at_the_floor_rather_than_raising():
    """A raise here would end the run's disk check as a crash; a duration
    that is not a number is a missing dimension."""
    for video_config in ({'duration': 'abc'}, {'duration': float('nan')}):
        step = _make_step(acquire='video', video_config=video_config)
        assert estimate_step_write_mb(step, sizing=a_frame_sizing(), video_fps=30.0) == 50


# --- the run's start ---------------------------------------------------------


def _prepare(executor, tmp_path, protocol, **overrides):
    kwargs = {
        'protocol': protocol,
        'run_mode': SequencedCaptureRunMode.FULL_PROTOCOL,
        'sequence_name': 'write_size',
        'image_capture_config': _make_image_capture_config(),
        'autogain_settings': _make_autogain_settings(),
        'parent_dir': tmp_path,
    }
    kwargs.update(overrides)
    return executor.prepare(**kwargs)


def _a_run_no_disk_holds():
    """One still a scan, every minute for a million hours: no disk holds the
    run, and any disk holds one scan."""
    protocol = _make_single_step_protocol()
    protocol.modify_time_params(
        period=datetime.timedelta(minutes=1), duration=datetime.timedelta(hours=1_000_000)
    )
    return protocol


def test_a_run_too_big_for_its_disk_is_refused_at_its_start(executor, tmp_path):
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, tmp_path, _a_run_no_disk_holds())

    assert refusal.value.reason == 'disk_space_insufficient'
    message = refusal.value.message
    assert '60,000,000 scans' in message, message
    assert 'is free at' in message and str(tmp_path) in message, message


def test_a_run_whose_drive_cannot_be_read_is_refused_at_its_start(executor, tmp_path, monkeypatch):
    # The folder is there and the drive under it does not answer: nothing
    # the run wrote there could be kept, so it does not start unguarded.
    free_space_unreadable(monkeypatch)
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, tmp_path, _make_single_step_protocol())

    assert refusal.value.reason == 'capture_location_unusable'
    assert str(tmp_path) in refusal.value.message, refusal.value.message


def test_a_scan_of_the_same_protocol_is_sized_at_one_scan(executor, tmp_path):
    plan = _prepare(
        executor,
        tmp_path,
        _a_run_no_disk_holds(),
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        max_scans=1,
    )

    assert plan.frame_sizing is not None


def test_a_run_that_saves_no_images_is_not_sized(executor, tmp_path):
    plan = _prepare(executor, tmp_path, _a_run_no_disk_holds(), enable_image_saving=False)

    assert plan.frame_sizing is None


def test_a_run_whose_camera_frame_size_is_unknown_is_refused(executor, tmp_path, monkeypatch):
    # a stand-in by design: a camera brought up without reporting its frame
    # size is a fault the simulator cannot yet produce (bring-up fills it).
    monkeypatch.setattr(type(executor._scope.camera), 'frame_size', property(lambda self: None))

    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, tmp_path, _make_single_step_protocol())

    assert refusal.value.reason == 'frame_size_unknown'


# --- the per-write check -----------------------------------------------------


def _write_one(writer, tmp_path, step, enable_image_saving=True):
    from _lumascope.modules.protocol_image_writer import CapturedFrame

    return writer.write_capture(
        step_index=0,
        scan_count=0,
        enable_image_saving=enable_image_saving,
        captured_image=CapturedFrame(
            still=CapturedStill(image=np.zeros((4, 4), dtype=np.uint8), record=frame_record()),
            objective_id='4x Oly',
            position=unpositioned(),
            fallback_focus=None,
        ),
        step=step,
        name='stepA_BF',
        save_folder=str(tmp_path),
        output_format='TIFF',
    )


def test_a_failure_of_the_estimate_is_raised_not_written_past(tmp_path, monkeypatch):
    """The probe's own failure (an ``OSError``) lets the write proceed; a
    failure of LumaViewPro's own estimate is raised where it happens instead
    of switching the disk guard off."""
    writer = _bare_protocol_writer()
    saves = []
    monkeypatch.setattr(
        '_lumascope.modules.protocol_image_writer.save_image',
        lambda scope, **kwargs: saves.append(kwargs) or (tmp_path / 'out.tiff'),
    )

    unsized = _protocol_step()
    del unsized['Sum']
    with pytest.raises(KeyError):
        _write_one(writer, tmp_path, unsized)
    assert not saves

    def unreadable(folder, needed):
        raise OSError('the folder cannot be read')

    monkeypatch.setattr('_lumascope.modules.common_utils.check_disk_space_ok', unreadable)
    _write_one(writer, tmp_path, _protocol_step())
    assert len(saves) == 1


def test_a_write_that_saves_no_image_is_not_sized(tmp_path, monkeypatch):
    """A run with saving off (the GUI's toggle, REST's ``enable_image_saving``)
    still writes its record into the run folder; ``prepare()`` sized none of
    it, so the per-write check asks nothing either."""
    writer = _bare_protocol_writer(frame_sizing=None)
    probed = []
    monkeypatch.setattr(
        '_lumascope.modules.common_utils.check_disk_space_ok',
        lambda folder, needed: probed.append(needed) or (True, 1e9),
    )

    _write_one(writer, tmp_path, _protocol_step(), enable_image_saving=False)

    assert not probed
