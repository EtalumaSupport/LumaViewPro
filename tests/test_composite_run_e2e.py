# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A headless composite run, end to end, against real files.

Every other test of the composite run kind stops short of the thing that
actually has to work. The config tests assemble a plan against a mocked
engine; the outcome tests drive the state machine directly; the headless
merge test calls ``CompositeGeneration`` on files it wrote itself. None of
them exercises the WIRING -- the run loop reaching the image writer, the
writer's per-run still count reaching the merge's drain wait, and the merge
reaching the outcome the caller is blocked on. That wiring is precisely
what a mock cannot check, because a mock is what stands in for it.

So this runs the real thing: a simulated scope, the production executor
topology, ``enable_image_saving=True``, real TIFFs on a real disk, and the
real merge. The assertion that matters is the humblest one -- the path
``run_composite`` hands back is a file that exists and opens as an image.

One caveat is worth stating, because it was measured rather than assumed:
on simulated hardware the writes land well before cleanup runs, so the
merge's drain wait is not load-bearing here by default -- deleting it
leaves the rest of this file green. Only the test that delays the write
holds that wait down.
"""

import contextlib
import pathlib
import time

import pytest
import tifffile as tf

from modules.image_mode import OUTPUT_FORMAT_TIFF
from tests.scope_fakes import home_sim_scope
from tests.test_composite_run_config import _settings as _base_settings

# Two channels is the minimum a merge can consume, and one of them is
# transmitted: that is the pairing whose blend actually reads a threshold,
# so a threshold that never reached the merge fails here rather than
# passing quietly on a fluorescence-only stack.
_ACQUIRING = ('BF', 'Blue')


def headless_settings(
    tmp_path,
    acquiring=_ACQUIRING,
    sequenced_format=OUTPUT_FORMAT_TIFF,
    live_format=OUTPUT_FORMAT_TIFF,
):
    """The composite config fixture, plus what a real RUN needs on top.

    ``test_composite_run_config`` owns the layer shape; assembly is all it
    needs. A run also has to know where to write, where the stage is
    relative to the plate, and what each layer's blend threshold is.

    The channel set and the two output formats are parameters because the
    other headless composite files vary exactly those: which channels
    light, how few is too few, and which preference the composite follows.
    """
    settings = _base_settings(acquiring=acquiring, sequenced_format=sequenced_format)
    settings['image_output_format']['live'] = live_format
    for layer in settings:
        if isinstance(settings[layer], dict) and 'acquire' in settings[layer]:
            settings[layer]['composite_brightness_threshold'] = 25
    settings['live_folder'] = str(tmp_path)
    settings['stage_offset'] = {'x': 0.0, 'y': 0.0}
    settings['turret_objectives'] = {}
    return settings


def _complete_the_bring_up(session, settings):
    """Finish what ``ScopeSession.create_headless`` leaves undone.

    THIS FIXTURE IS A DEFECT MARKER, not a convenience. Every call below is
    production code that a GUI session runs during bring-up and a headless
    one never does:

    - ``create`` constructs the three loaders under individual try/except
      with a notification each; ``create_headless`` constructs none, so its
      session hands ``None`` to the engine and an L2 caller gets
      ``AttributeError`` instead of a typed refusal.
    - ``Lumascope.initialize`` -- documented "call once after construction"
      -- has two production callers, both in ``ui/microscope_settings.py``.
      Without it the scope carries no objective, and every image write
      fails with ``ConfigError: Objective not set`` on the writer thread,
      so a headless run captures and saves nothing.

    When that bring-up moves where it belongs, this function's body becomes
    dead and the fixture below should call ``create_headless`` alone. Until
    then, doing it here is what lets this test exercise the composite
    wiring rather than re-discover a session-composition gap.
    """
    from modules import coord_transformations, labware_loader, objectives_loader
    from modules.scope_init_config import ScopeInitConfig

    session.objective_helper = objectives_loader.ObjectiveLoader(source_path='.')
    session.wellplate_loader = labware_loader.WellPlateLoader(source_path='.')
    session.coordinate_transformer = coord_transformations.CoordinateTransformer()
    # The engine took its copies at construction, before this ran.
    session.sequenced_capture_runner._wellplate_loader = session.wellplate_loader
    session.sequenced_capture_runner._coordinate_transformer = session.coordinate_transformer

    plate = session.wellplate_loader.get_plate(plate_key=settings['protocol']['labware'])
    session.scope.initialize(ScopeInitConfig.from_settings(settings, plate))


@contextlib.contextmanager
def open_composite_session(settings):
    """A headless session on simulated hardware, ready to run a composite.

    Yields ``(session, runner)`` with the executors started and the scope
    homed, and tears both down on exit. The other headless composite files
    build their own settings and open the session through this, so there
    is exactly one description of what a ready headless scope is.
    """
    from modules.scope_session import ScopeSession

    session = ScopeSession.create_headless(settings=settings)
    _complete_the_bring_up(session, settings)

    scope = session.scope
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    home_sim_scope(scope)

    session.start_executors()
    runner = session.create_protocol_runner()
    try:
        yield session, runner
    finally:
        runner.shutdown()
        session.shutdown()


@pytest.fixture
def composite_session(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        yield session, runner, tmp_path


def single_run_dir(tmp_path):
    """The single timestamped run directory the run created."""
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f'expected exactly one run directory, found {dirs}'
    return dirs[0]


class TestCompositeRunEndToEnd:
    """One composite run, from the L2 call to the file it produced."""

    def test_the_returned_path_is_a_readable_composite(self, composite_session):
        _session, runner, tmp_path = composite_session

        artifact = runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path))

        assert artifact, 'run_composite returned no path for a run that succeeded'
        path = pathlib.Path(artifact)
        assert path.exists(), (
            f'run_composite reported {path}, which does not exist -- the '
            f"return is the caller's only evidence the merge produced anything"
        )
        merged = tf.imread(str(path))
        assert merged.size > 0, 'the merged composite opened but carries no pixels'
        # The merge blends the channels into one colour image; a merged
        # artifact that came back single-channel means the blend never ran.
        assert merged.ndim == 3, f'expected a colour composite, got shape {merged.shape}'

    def test_the_per_channel_frames_reached_disk(self, composite_session):
        # The merge reads its inputs back off disk, so the run's own writes
        # are part of the contract, not an implementation detail.
        _session, runner, tmp_path = composite_session

        runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path))

        run_dir = single_run_dir(tmp_path)
        frames = sorted(p.name for p in run_dir.glob('*.tiff'))
        assert len(frames) == len(_ACQUIRING), (
            f'expected one frame per acquiring channel {_ACQUIRING}, found {frames}'
        )
        for channel in _ACQUIRING:
            assert any(channel in name for name in frames), (
                f'no saved frame names channel {channel}: {frames}'
            )

    def test_the_composite_lands_under_the_run_directory(self, composite_session):
        # A merged artifact written outside the run it belongs to is how a
        # composite ends up beside the NEXT run's frames.
        _session, runner, tmp_path = composite_session

        artifact = pathlib.Path(runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path)))

        assert single_run_dir(tmp_path) in artifact.parents, (
            f'{artifact} is not inside the run directory that produced it'
        )

    def test_the_run_is_fully_settled_when_the_call_returns(self, composite_session):
        # run_composite blocks on the merge, so by the time it returns
        # nothing of this run may still be in flight -- an L2 caller's next
        # run would otherwise be refused for writes it was never told about.
        session, runner, tmp_path = composite_session

        runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path))

        assert not session.is_protocol_running, (
            'run_composite returned while the run still held the activity claim'
        )
        outcome = runner._executor.merge_outcome()
        settled = outcome.wait(timeout_s=0)
        assert settled is not None and settled.merged, (
            f'the merge outcome was not resolved-merged at return: {settled}'
        )

    def test_the_merge_waits_for_the_writes_it_will_read(self, composite_session, monkeypatch):
        # THE reason this file exists. The merge reads its inputs off disk,
        # so it must not start until this run's stills have landed -- and
        # nothing else here proves that: on simulated hardware the writes
        # finish long before cleanup, so the drain wait is never actually
        # load-bearing and deleting it outright leaves every other test in
        # this file green (measured, not assumed).
        #
        # Delaying the write is what makes the race observable: the merge
        # now reaches its wait with the frames still in flight, so a merge
        # that skips the wait finds a directory holding fewer channels than
        # the run captured -- which is a composite silently missing a
        # channel, the failure the bound exists to prevent.
        import modules.protocol_image_writer as piw

        real_save = piw.save_image

        def _slow_save(*args, **kwargs):
            time.sleep(0.6)
            return real_save(*args, **kwargs)

        monkeypatch.setattr(piw, 'save_image', _slow_save)

        _session, runner, tmp_path = composite_session
        artifact = pathlib.Path(runner.run_composite(sequence_name='e2e', parent_dir=str(tmp_path)))

        run_dir = single_run_dir(tmp_path)
        frames = sorted(p.name for p in run_dir.glob('*.tiff'))
        assert len(frames) == len(_ACQUIRING), (
            f'the merge consumed a directory still filling: {frames}'
        )
        assert artifact.exists(), (
            'the merge produced no artifact from frames that were still being written'
        )

    def test_a_second_composite_runs_after_the_first(self, composite_session):
        # The per-run write counter and the outcome object are both
        # per-run: a first run that left either armed would refuse or hang
        # the second, and only a real run exercises that reset.
        _session, runner, tmp_path = composite_session

        first = pathlib.Path(runner.run_composite(sequence_name='e2e_1', parent_dir=str(tmp_path)))
        second = pathlib.Path(runner.run_composite(sequence_name='e2e_2', parent_dir=str(tmp_path)))

        assert first != second, 'the second composite overwrote the first'
        assert first.exists() and second.exists(), (
            f'both composites must survive: {first.exists()}, {second.exists()}'
        )


class TestStartComposite:
    """The non-blocking half: assembly and launch without the wait."""

    def test_the_trigger_token_is_the_callers(self, composite_session):
        # A constant here instead of a parameter would make a GUI click
        # during an API composite read as that run's OWN second click, so
        # the click would abort someone else's run instead of being refused.
        _session, runner, tmp_path = composite_session

        outcome = runner.start_composite(
            sequence_name='start_token',
            parent_dir=str(tmp_path),
            run_trigger_source='composite',
        )
        assert outcome.wait(timeout_s=120) is not None, 'the run never settled'
        assert runner.run_trigger_source() == 'composite'

    def test_the_api_entry_point_keeps_its_own_token(self, composite_session):
        _session, runner, tmp_path = composite_session

        runner.run_composite(sequence_name='api_token', parent_dir=str(tmp_path))

        assert runner.run_trigger_source() == 'api_composite'

    def test_each_run_gets_the_outcome_it_started(self, composite_session):
        # The outcome used to be read back off the executor after start
        # returned. A run that fails at start releases its activity claim
        # synchronously, so a rival can commit in between and the caller
        # ends up waiting on the rival's run. Handing the object back from
        # start is what makes that unrepresentable.
        _session, runner, tmp_path = composite_session

        first = runner.start_composite(sequence_name='own_1', parent_dir=str(tmp_path))
        first_settled = first.wait(timeout_s=120)
        second = runner.start_composite(sequence_name='own_2', parent_dir=str(tmp_path))
        second_settled = second.wait(timeout_s=120)

        assert first is not second, "the second run reused the first run's outcome object"
        assert first_settled.merged and second_settled.merged, (
            f'both merges must succeed: {first_settled}, {second_settled}'
        )
        assert first_settled.artifact_path != second_settled.artifact_path


class TestTheEngineMatchesTheWorkerItReplaces:
    """The run kind's merged output equals what the GUI worker produced.

    The worker is being deleted, so the question is whether anything about
    the image changes for a user who clicks the same button. Both paths turn
    out to funnel through the SAME build_composite -- the blend was never
    duplicated -- so what is actually being compared here is the two
    orchestrators' INPUT assembly: which frames are chosen, at what depth,
    and with which thresholds. That is where a divergence could hide, and it
    would show up as different pixels.

    Array equality, not a visual check: a threshold applied on the wrong
    scale still looks like a composite.
    """

    def test_the_merged_array_is_identical_to_the_workers(self, composite_session):
        import numpy as np

        from modules.composite_builder import build_composite

        _session, runner, tmp_path = composite_session

        artifact = pathlib.Path(
            runner.run_composite(sequence_name='equiv', parent_dir=str(tmp_path))
        )
        run_dir = single_run_dir(tmp_path)

        frames = {}
        for path in run_dir.glob('*.tiff'):
            for channel in _ACQUIRING:
                if channel in path.name:
                    frames[channel] = tf.imread(path)
        assert set(frames) == set(_ACQUIRING), f'missing per-channel frames: {sorted(frames)}'

        transmitted = frames['BF']
        fluorescence = {name: arr for name, arr in frames.items() if name != 'BF'}

        # The worker read its threshold as an absolute value on the OUTPUT
        # 8-bit scale, and the settings carry a percentage, so the conversion
        # is part of what is being compared.
        thresholds = dict.fromkeys(fluorescence, 25 / 100 * 255)

        expected = build_composite(
            channel_images=fluorescence,
            significant_bits=8 if transmitted.dtype == np.uint8 else 12,
            transmitted_image=transmitted,
            brightness_thresholds=thresholds,
        )

        assert np.array_equal(tf.imread(artifact), expected), (
            'the run kind produced a different composite than build_composite '
            'does from the same on-disk frames; the orchestrators disagree '
            'about depth, channel selection, or threshold scale'
        )
