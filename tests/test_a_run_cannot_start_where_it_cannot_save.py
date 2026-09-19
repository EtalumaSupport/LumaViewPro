# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run whose save location cannot take its output is refused, not failed.

The save-folder check lived in a Kivy widget method, so a script, the SDK
and REST never met it: their run committed, moved the stage, took images
and then discovered there was nowhere to put them -- a failed run for a
request that could have been turned away.

The widget also answered the question by WRITING: it created a hardcoded
ProtocolData path and touched a file in it. The engine cannot do that --
prepare() promises a refused request leaves nothing on disk -- so the
predicate is a non-writing os.access on the nearest existing ancestor of
the directory the run was actually handed.

That is a weaker predicate than a write, and deliberately so: a full disk
and a mount that lies about permissions still reach the allocator at
commit time and are reported there. What it buys is that the FIVE
starters are checked instead of one, and each against its own folder.

The behaviour moves in both directions, which is why both are pinned
here. The z-stack, composite and standalone-autofocus starters gain a
refusal they never had, because the widget only ever probed ProtocolData.
The autofocus scan loses one it had, because it passes no directory and
writes nothing -- the widget refused it over a folder that run never
touches.
"""

from __future__ import annotations

import os
import sys

import pytest

from modules import path_utils
from modules.exceptions import ProtocolRunRefusedError
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.protocol_drives import autofocus_snapshot
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    executor,
    executors,
    scope,
)


# A chmod is only a permission test for someone the permissions apply to:
# root bypasses the mode bits, and Windows does not express "no write" as
# one. The file-in-the-path case below covers the same refusal everywhere.
needs_posix_permissions = pytest.mark.skipif(
    sys.platform == 'win32' or os.geteuid() == 0,
    reason='directory mode bits only refuse a non-root POSIX user',
)


def _prepare(executor, tmp_path, parent_dir, **overrides):
    kwargs = {
        'protocol': _make_single_step_protocol(),
        'run_trigger_source': 'zstack',
        'run_mode': SequencedCaptureRunMode.SINGLE_SCAN,
        'sequence_name': 'capture_location',
        'image_capture_config': _make_image_capture_config(),
        'autogain_settings': _make_autogain_settings(),
        'autofocus_snapshot': autofocus_snapshot(),
        'parent_dir': parent_dir,
        'max_scans': 1,
        'callbacks': {},
    }
    kwargs.update(overrides)
    return executor.prepare(**kwargs)


class TestTheEngineRefuses:
    @needs_posix_permissions
    def test_a_save_location_that_cannot_be_written_is_refused(self, executor, tmp_path):
        read_only = tmp_path / 'read_only'
        read_only.mkdir(mode=0o500)
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _prepare(executor, tmp_path, read_only / 'Manual' / 'Z-Stacks')
        finally:
            # Restored so the tmp_path teardown can remove it.
            read_only.chmod(0o700)

        assert refusal.value.reason == 'capture_location_unusable', (
            'the reason code is what a REST or SDK caller branches on'
        )

    def test_a_save_location_behind_a_file_is_refused(self, executor, tmp_path):
        """The configured save folder names something that is not a folder.

        Reachable without a chmod, so it is the case that runs on every
        platform the app ships to.
        """
        not_a_folder = tmp_path / 'live_folder'
        not_a_folder.write_text('this is a file')

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, tmp_path, not_a_folder / 'Manual' / 'Z-Stacks')

        assert refusal.value.reason == 'capture_location_unusable'

    def test_the_refusal_names_the_location_and_the_problem(self, executor, tmp_path):
        """A reason code alone sends the user looking for which folder.

        The widget's popup named the path; a caller cannot reconstruct it
        from the code, and neither can the user standing at the scope.
        """
        not_a_folder = tmp_path / 'live_folder'
        not_a_folder.write_text('this is a file')
        parent_dir = not_a_folder / 'Manual' / 'Z-Stacks'

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, tmp_path, parent_dir)

        assert str(parent_dir) in refusal.value.message, (
            f'the refusal must name the folder it could not use: {refusal.value.message!r}'
        )
        assert str(not_a_folder) in refusal.value.message, (
            f'the refusal must name what is actually wrong: {refusal.value.message!r}'
        )

    @needs_posix_permissions
    def test_a_run_that_suppresses_artifacts_is_still_checked(self, executor, tmp_path):
        """Suppressing artifacts does not mean writing nothing.

        An engineering-mode autofocus passes a real directory with
        disable_saving_artifacts set: the run directory is skipped, and
        the autofocus characterization CSV still lands in that directory.
        A run gated on the suppression flag would miss it.
        """
        read_only = tmp_path / 'read_only'
        read_only.mkdir(mode=0o500)
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _prepare(
                    executor,
                    tmp_path,
                    read_only / 'Autofocus Characterization',
                    run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                    run_trigger_source='autofocus',
                    disable_saving_artifacts=True,
                    save_autofocus_data=True,
                )
        finally:
            read_only.chmod(0o700)

        assert refusal.value.reason == 'capture_location_unusable'


class TestTheCasesThatMustNotBeRefused:
    def test_a_save_location_that_does_not_exist_yet_is_accepted(self, executor, tmp_path):
        """The normal first run: Manual/Z-Stacks has never been created.

        Testing the leaf itself would refuse every run into a fresh save
        location, which is every run on a new install.
        """
        plan = _prepare(executor, tmp_path, tmp_path / 'live' / 'Manual' / 'Z-Stacks')

        assert plan is not None
        executor.reset(requester='zstack')

    @needs_posix_permissions
    def test_a_run_that_saves_nowhere_is_not_refused(self, executor, tmp_path):
        """The refusal the autofocus scan LOSES, pinned deliberately.

        It passes no directory, so it writes nothing and there is nothing
        to check. The widget refused it whenever ProtocolData was
        unwritable -- a folder that run never touches.
        """
        read_only = tmp_path / 'read_only'
        read_only.mkdir(mode=0o500)
        try:
            plan = _prepare(
                executor,
                tmp_path,
                None,
                run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                run_trigger_source='autofocus_scan',
                enable_image_saving=False,
            )
        finally:
            read_only.chmod(0o700)

        assert plan is not None, 'a run with nowhere to save has no save location to refuse'
        executor.reset(requester='autofocus_scan')


class TestThePredicateDoesNotWrite:
    """prepare() states it writes nothing to disk; this is the part that could."""

    def test_the_check_leaves_nothing_behind(self, tmp_path):
        before = sorted(p.name for p in tmp_path.iterdir())

        assert path_utils.capture_location_problem(tmp_path / 'Manual' / 'Composites') is None

        assert sorted(p.name for p in tmp_path.iterdir()) == before, (
            'the predicate created something; the widget probe it replaces did '
            'exactly that, and prepare() promises a refused run is a no-op'
        )

    def test_a_refused_prepare_leaves_nothing_behind(self, executor, tmp_path):
        not_a_folder = tmp_path / 'live_folder'
        not_a_folder.write_text('this is a file')
        before = sorted(p.name for p in tmp_path.iterdir())

        with pytest.raises(ProtocolRunRefusedError):
            _prepare(executor, tmp_path, not_a_folder / 'Manual' / 'Z-Stacks')

        assert sorted(p.name for p in tmp_path.iterdir()) == before
