# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ScopeSession.last_save_folder`` is the folder the last save landed in, whoever asked.

A still, a run's images and a post-processing build each move it when their
files land, never when the work starts, so work that wrote nothing leaves it
where it was. Each save here is asked for through the session, as REST and
the SDK ask for it; the GUI's "Open last save folder" opens what it says.
A recording's case is the recording controller's own
(``test_manual_recording_controller.py``, ``TestTheSessionIsToldWhereItSaved``).
"""

from __future__ import annotations

import pathlib

import pytest

from _lumascope.modules.exceptions import PostProcessingFailedError

# The tiled run folder the build tests save into, as a run of this release saves it.
from tests.test_a_build_names_what_it_left_out import run_folder  # noqa: F401

RESULT_TIMEOUT_S = 30


def test_before_any_save_it_is_none_and_no_caller_can_write_it(sim_session, tmp_path):
    assert sim_session.last_save_folder is None
    with pytest.raises(AttributeError):
        sim_session.last_save_folder = tmp_path


def test_a_still_moves_it_to_the_stills_folder(sim_session):
    from tests.scope_fakes import home_sim_scope

    home_sim_scope(sim_session.scope)

    saved = sim_session.manual_capture.capture(false_color_on=False).result(
        timeout=RESULT_TIMEOUT_S
    )

    assert sim_session.last_save_folder == saved[0].file.parent


@pytest.fixture
def scanning(tmp_path):
    """A homed simulated session ready to run, and a one-step BF scan of it."""
    from tests.test_a_late_write_records_its_frame import _protocol, _step
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        yield session, runner, _protocol([_step('P1', 0, x=20.0, gain=1.0)])


def test_a_run_moves_it_to_its_folder_once_its_images_land(scanning, tmp_path):
    session, runner, protocol = scanning

    pending = runner.start_single_scan(protocol=protocol, parent_dir=str(tmp_path / 'runs'))
    assert pending.result(timeout=60).status == 'completed'
    assert pending.files.result(timeout=60) is not None, "the run's files never finished"

    run_dir = pending.progress().run_dir
    assert run_dir is not None
    assert session.last_save_folder == pathlib.Path(run_dir)


def test_a_run_that_saves_no_image_leaves_it(scanning, tmp_path):
    session, runner, protocol = scanning

    pending = runner.start_single_scan(
        protocol=protocol, parent_dir=str(tmp_path / 'runs'), enable_image_saving=False
    )
    assert pending.result(timeout=60).status == 'completed'
    pending.files.result(timeout=60)

    assert session.last_save_folder is None


def test_a_stitch_moves_it_to_the_subfolder_it_wrote(sim_session, run_folder):
    stitched = sim_session.post_processing.stitch(run_folder)

    assert stitched.new_count > 0
    assert sim_session.last_save_folder == stitched.artifact_paths[-1].parent
    assert sim_session.last_save_folder != run_folder, 'a stitch writes under its own subfolder'


def test_a_stitch_that_makes_nothing_new_leaves_it(sim_session, run_folder):
    """A still saved between the two stitches lands outside the run folder, so
    the second stitch moving the record to the folder it read would show."""
    from tests.scope_fakes import home_sim_scope

    home_sim_scope(sim_session.scope)
    sim_session.post_processing.stitch(run_folder)
    saved = sim_session.manual_capture.capture(false_color_on=False).result(
        timeout=RESULT_TIMEOUT_S
    )
    still_folder = saved[0].file.parent
    assert run_folder not in (still_folder, *still_folder.parents)

    again = sim_session.post_processing.stitch(run_folder)

    assert again.new_count == 0
    assert sim_session.last_save_folder == still_folder


def test_a_quick_enhance_moves_it_to_the_folder_it_wrote(sim_session, run_folder):
    enhanced = sim_session.post_processing.enhance(run_folder)

    assert enhanced.created
    assert sim_session.last_save_folder == enhanced.output_folder


def test_a_stitch_that_fails_part_way_moves_it_to_what_it_did_write(
    sim_session,
    run_folder,
    monkeypatch,
):
    from _lumascope.modules.protocol_post_processing_result import PostProcResult
    from _lumascope.modules.stitcher import Stitcher

    real = Stitcher._group_algorithm
    groups = []

    def first_group_fails(self, **kwargs):
        groups.append(kwargs)
        if len(groups) == 1:
            return PostProcResult.failed('no tiles')
        return real(self, **kwargs)

    # a stand-in by design: the simulator cannot yet make a stitch group fail.
    monkeypatch.setattr(Stitcher, '_group_algorithm', first_group_fails)

    with pytest.raises(PostProcessingFailedError) as failed:
        sim_session.post_processing.stitch(run_folder)

    (written,) = failed.value.produced_paths
    assert sim_session.last_save_folder == pathlib.Path(written).parent


def test_the_button_opens_the_last_save_and_the_live_folder_before_any(
    gui_ctx,
    run_folder,
    monkeypatch,
):
    import ui.post_processing as gui_post_processing

    opened = []
    monkeypatch.setattr(
        gui_post_processing.subprocess, 'Popen', lambda command: opened.append(command[-1])
    )
    live = str(pathlib.Path(gui_ctx.settings['live_folder']).resolve())

    gui_post_processing.open_last_save_folder()
    stitched = gui_ctx.session.post_processing.stitch(run_folder)
    gui_post_processing.open_last_save_folder()

    assert opened == [live, str(stitched.artifact_paths[-1].parent)]
