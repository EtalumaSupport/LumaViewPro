# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A folder build takes a run's TIFF or OME-TIFF images, and refuses a JPG run for its format.

The owner is ``ProtocolPostProcessor.load_folder``: its format guard is the
one statement that a build takes TIFF, and its ``no_images`` the one answer
for a folder with nothing to build from. Every test runs Z-Projection,
Stitch, Composite and Video, through the session, on run folders the
simulator saved, since the walk of the folder is half of what is pinned.
"""

from __future__ import annotations

import logging
import shutil

import cv2
import numpy as np
import pytest

from _lumascope.modules.exceptions import PostProcessingRefusedError
from _lumascope.modules.image_mode import OutputFormat
from _lumascope.modules.scope_session import ScopeSession
from tests.count_runs import _over_a_well, _saved
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

_BUILDERS = ('zproject', 'stitch', 'composite', 'video')


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """A session, and its runs by name: ``jpg``, one BF and one LED 0 still,
    as JPG; ``<format>_frames``, one BF still and an LED 0 recording saved as
    frames, for JPG, TIFF and OME-TIFF; ``unsaved``, a run with saving off."""
    live = tmp_path_factory.mktemp('runs')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        home_sim_scope(session.scope)
        runner = session.protocol_runner()

        def run(name: str, fmt: OutputFormat, save: bool = True):
            session.update_settings('image_output_format.sequenced', fmt)
            config = session.get_sequenced_capture_config(tiling='1x1', use_zstacking=False)
            config['positions'] = _over_a_well(session)
            protocol = session.scope.protocols.create_protocol(input_config=config)
            return _saved(
                live,
                lambda: runner.start_single_scan(
                    protocol=protocol,
                    sequence_name=name,
                    parent_dir=str(live),
                    enable_image_saving=save,
                ),
            )

        session.set_channel_settings('BF', acquire='image')
        session.set_channel_settings(LED0, acquire='image')
        saved = {
            'jpg': run('jpg', OutputFormat.JPG),
            'unsaved': run('unsaved', OutputFormat.TIFF, save=False),
        }
        session.set_channel_settings(LED0, acquire='video')
        session.update_settings('video_as_frames', True)
        for fmt in (OutputFormat.JPG, OutputFormat.TIFF, OutputFormat.OME_TIFF):
            saved[f'{fmt.name.lower()}_frames'] = run(f'frames_{fmt.name}', fmt)
        yield session, saved
    finally:
        session.shutdown()


def _copy(runs, name: str, tmp_path):
    own = tmp_path / name
    shutil.copytree(runs[1][name], own)
    return own


def _recording(folder):
    (recording,) = [d for d in folder.iterdir() if d.is_dir()]
    return recording


def _answers(session, folder) -> dict:
    """Each builder's answer on *folder*: the refusal's reason and words, or the files it built."""
    answers = {}
    for name in _BUILDERS:
        try:
            built = getattr(session.post_processing, name)(folder)
        except PostProcessingRefusedError as refused:
            answers[name] = (refused.reason, str(refused))
        else:
            answers[name] = ('built', sorted(p.name for p in built.artifact_paths))
    return answers


def _refused_for(answers: dict, jpg: str) -> dict:
    """The builders whose answer is not the format refusal naming *jpg* first."""
    return {
        name: answer
        for name, answer in answers.items()
        if answer[0] != 'unsupported_source_format'
        or f'First unsupported JPG file: {jpg}.' not in answer[1]
    }


def test_a_jpg_run_is_refused_for_its_format_by_every_builder(runs, tmp_path):
    folder = _copy(runs, 'jpg', tmp_path)
    first = sorted(p.name for p in folder.iterdir() if p.suffix == '.jpg')[0]

    assert _refused_for(_answers(runs[0], folder), first) == {}


def test_a_jpg_run_with_a_frames_recording_is_refused_and_its_recording_builds(runs, tmp_path):
    # The recording's frames are TIFF and the still is JPG: the run folder is
    # refused for the still, never built from the recording alone.
    folder = _copy(runs, 'jpg_frames', tmp_path)
    (still,) = [p.name for p in folder.iterdir() if p.suffix == '.jpg']

    assert _refused_for(_answers(runs[0], folder), still) == {}
    video = runs[0].post_processing.video(_recording(folder))
    assert [p.suffix for p in video.artifact_paths] == ['.mp4']


def _drop_images_named_like_frames(folder):
    """A PNG and a JPG in the recording's folder, each named like one of its frames."""
    recording = _recording(folder)
    frame = sorted(recording.iterdir())[0].name.split('.')[0]
    blank = np.zeros((32, 32), np.uint8)
    cv2.imwrite(str(recording / f'{frame}.png'), blank)
    cv2.imwrite(str(recording / f'{frame.replace("_0000", "_9999")}.jpg'), blank)


# parametrize-justified: each format is its own saved run
@pytest.mark.parametrize('run', ['tiff_frames', 'ome_tiff_frames'])
def test_an_image_named_like_a_frame_changes_no_build(run, runs, tmp_path):
    untouched = _copy(runs, run, tmp_path / 'untouched')
    dropped = _copy(runs, run, tmp_path / 'dropped')
    _drop_images_named_like_frames(dropped)

    # The recording folders are picked before any build writes a folder of its own.
    for picked_untouched, picked_dropped in [
        (_recording(untouched), _recording(dropped)),
        (untouched, dropped),
    ]:
        today = _answers(runs[0], picked_untouched)
        assert today['video'][0] == 'built', today
        assert _answers(runs[0], picked_dropped) == today


def test_a_run_that_saved_nothing_is_refused_no_images_and_logs_no_error(runs, tmp_path, caplog):
    folder = _copy(runs, 'unsaved', tmp_path)

    with caplog.at_level(logging.DEBUG):
        answers = _answers(runs[0], folder)

    assert {name: answer[0] for name, answer in answers.items()} == dict.fromkeys(
        _BUILDERS, 'no_images'
    )
    assert all('No image files were found in the selected folder' in a[1] for a in answers.values())
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_a_jpg_run_missing_a_recorded_jpg_is_refused_for_the_ones_present(runs, tmp_path):
    folder = _copy(runs, 'jpg', tmp_path)
    gone, present = sorted(p for p in folder.iterdir() if p.suffix == '.jpg')
    gone.unlink()

    assert _refused_for(_answers(runs[0], folder), present.name) == {}
