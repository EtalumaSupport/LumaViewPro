# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the LEDs do across a composite run, measured on the board.

The composite run kind is the one run kind a user triggers while standing
at the scope with the live preview lit, so its illumination contract is
different from a scan's: it borrows the LEDs and hands them back the way
it found them (``leds_state_at_end='return_to_original'``), and only a
FATAL abort forces the sample dark. Every other file about this run kind
checks what reached DISK. Nothing checks what reached the SAMPLE.

That gap matters because the illumination decisions are spread across
three layers -- the config picks the channel set, the step runner turns
each step into a STEP_LIGHT / STEP_BOUNDARY pair, and the authority
diffs those against the cached state -- and none of them can be checked
from the merged artifact. A run that lit two channels at once, left one
burning, or extinguished the user's live channel produces exactly the
same composite as a correct one.

So the instrument here is the simulated LED board itself: observation
wrappers around the driver's write methods record every command in
order, the driver's ``_channel_states`` says what is lit when the dust
settles, and a wrapper around the image writer's ``capture`` marks where
each frame was grabbed inside that sequence. Nothing is mocked -- every
wrapper records and then calls through to the real method.
"""

import pathlib
import threading

import pytest

import modules.protocol_image_writer as protocol_image_writer
from modules.config_helpers import get_composite_channels
from tests.test_composite_run_e2e import (
    headless_settings,
    open_composite_session,
    single_run_dir,
)

# Lit before the run and never acquired: the channel whose survival is
# the whole question. A channel the run itself lights would be restored
# by accident, so the pre-run state has to be one the run has no reason
# to touch.
_PRE_RUN_CHANNEL = 'Red'
_PRE_RUN_MA = 50.0


def _light_pre_run_channel(session):
    """Light the live channel a user would have on before clicking."""
    session.scope.illumination.led_on(_PRE_RUN_CHANNEL, _PRE_RUN_MA)


def _lit_channels(session):
    """Colour -> mA for every channel the simulated board has driven on."""
    driver = session.scope._led_driver
    return {driver.ch2color(ch): ma for ch, ma in driver._channel_states.items() if ma and ma > 0}


def _record_led_commands(monkeypatch, session):
    """Record every LED write the run issues, in order, and call through.

    An observation wrapper, not a double: each recorded method invokes the
    real driver method, so the board's state is exactly what an unwrapped
    run would have left. Both the blocking and the write-only fast
    variants are wrapped, because which one a path takes is an
    implementation detail this file must not depend on.
    """
    driver = session.scope._led_driver
    events = []

    def _wrap_on(name):
        original = getattr(driver, name)

        def recorded(channel, mA, *args, **kwargs):
            events.append(('on', driver.ch2color(channel), float(mA)))
            return original(channel, mA, *args, **kwargs)

        monkeypatch.setattr(driver, name, recorded)

    def _wrap_off(name):
        original = getattr(driver, name)

        def recorded(channel, *args, **kwargs):
            events.append(('off', driver.ch2color(channel), 0.0))
            return original(channel, *args, **kwargs)

        monkeypatch.setattr(driver, name, recorded)

    def _wrap_all_off(name):
        original = getattr(driver, name)

        def recorded(*args, **kwargs):
            events.append(('all_off', None, 0.0))
            return original(*args, **kwargs)

        monkeypatch.setattr(driver, name, recorded)

    for name in ('led_on', 'led_on_fast'):
        _wrap_on(name)
    for name in ('led_off', 'led_off_fast'):
        _wrap_off(name)
    for name in ('leds_off', 'leds_off_fast'):
        _wrap_all_off(name)
    return events


def _record_captures(monkeypatch, events):
    """Mark each frame grab in the same ordered list as the LED writes.

    ``ProtocolImageWriter.capture`` runs synchronously on the protocol
    worker between that step's STEP_LIGHT and its STEP_BOUNDARY, so a
    marker recorded here lands between the on and the off of the channel
    it belongs to. The eventual disk WRITE is asynchronous and would say
    nothing about ordering.
    """
    original = protocol_image_writer.ProtocolImageWriter.capture

    def recorded(self, *args, **kwargs):
        events.append(('capture', kwargs['step']['Color'], 0.0))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(protocol_image_writer.ProtocolImageWriter, 'capture', recorded)


def _lit_sequence(events):
    """The channels that were driven on, in order, without repeats."""
    order = []
    for kind, color, illumination_ma in events:
        if kind == 'on' and illumination_ma > 0 and color not in order:
            order.append(color)
    return order


def _frame_channels(run_dir, channels):
    """The subset of *channels* that has a saved frame in *run_dir*."""
    names = [p.name for p in run_dir.glob('*.tiff')]
    return {channel for channel in channels if any(channel in name for name in names)}


# ---------------------------------------------------------------------------
# A user stop mid-composite
# ---------------------------------------------------------------------------


_ABORT_ACQUIRING = ('BF', 'Blue', 'Green')


@pytest.fixture
def aborted_composite(tmp_path):
    """One composite stopped from a per-step callback after step one.

    The abort is fired from ``update_step_number``, which the step runner
    invokes as it advances off a completed step -- so the run is
    genuinely mid-sequence, with channels still uncaptured, rather than
    an abort racing a run that had already finished. ``reset()`` is
    documented non-blocking for its caller, so calling it from the
    protocol worker signals the unwind rather than running it inline.

    Yields the observations the assertions below read: the outcome, the
    step number the abort was fired at, the frames that reached disk, and
    the board state afterwards.
    """
    settings = headless_settings(tmp_path, acquiring=_ABORT_ACQUIRING)
    with open_composite_session(settings) as (session, runner):
        _light_pre_run_channel(session)
        pre_run = _lit_channels(session)

        aborted_at = []
        fired = threading.Event()

        def _abort_after_the_first_step(step):
            if aborted_at:
                return
            aborted_at.append(step)
            runner.abort()
            fired.set()

        outcome = runner.start_composite(
            sequence_name='abort',
            parent_dir=str(tmp_path),
            callbacks={'update_step_number': _abort_after_the_first_step},
            run_trigger_source='composite',
        )
        settled = outcome.wait(timeout_s=120)
        assert runner.wait_for_run_idle(timeout_s=60), 'the aborted run never went idle'

        yield {
            'session': session,
            'settled': settled,
            'aborted_at': aborted_at,
            'fired': fired.is_set(),
            'frames': sorted(p.name for p in tmp_path.rglob('*.tiff')),
            'pre_run': pre_run,
            'lit': _lit_channels(session),
        }


class TestAbortMidComposite:
    """A second click stops the run and hands the illumination back."""

    def test_the_abort_landed_before_every_channel_was_captured(self, aborted_composite):
        # Without this the rest of the class is vacuous: an abort that
        # arrives after the last step measures a NORMAL end-of-run
        # restore and would stay green even if the abort path forced the
        # sample dark.
        assert aborted_composite['fired'], 'the per-step callback never fired, so nothing aborted'
        assert aborted_composite['aborted_at'] == [2], (
            f'the abort was not fired at the first step boundary: {aborted_composite["aborted_at"]}'
        )
        assert len(aborted_composite['frames']) < len(_ABORT_ACQUIRING), (
            f'the run captured every channel before the abort landed: {aborted_composite["frames"]}'
        )

    def test_the_outcome_resolves_not_merged_as_aborted(self, aborted_composite):
        # A caller blocked on the outcome has to learn WHY no artifact
        # came back; an aborted run that resolved merged, or that never
        # resolved at all, leaves an L2 caller waiting on the bound.
        settled = aborted_composite['settled']
        assert settled is not None, 'the aborted run never settled its merge outcome'
        assert not settled.merged, f'an aborted run reported a merged artifact: {settled}'
        assert settled.artifact_path is None, f'an aborted run named an artifact: {settled}'
        assert settled.reason == 'aborted', (
            f"the abort reported reason {settled.reason!r}, not 'aborted'"
        )

    def test_the_run_releases_the_scope(self, aborted_composite):
        # A stop that leaves the activity claim held refuses every later
        # run and recording with nothing to point at.
        assert not aborted_composite['session'].is_protocol_running, (
            'the aborted run still holds the activity claim'
        )

    def test_the_pre_run_channel_is_lit_again_at_its_own_current(self, aborted_composite):
        # THE item. A user stop is not a fatal abort: the fatal path
        # forces every channel dark on purpose, and reusing that path for
        # a user stop would leave the live preview black after a Stop
        # click, with no error to explain it.
        assert aborted_composite['pre_run'] == {_PRE_RUN_CHANNEL: _PRE_RUN_MA}, (
            f'the pre-run setup did not light {_PRE_RUN_CHANNEL}: {aborted_composite["pre_run"]}'
        )
        assert aborted_composite['lit'].get(_PRE_RUN_CHANNEL) == _PRE_RUN_MA, (
            f'{_PRE_RUN_CHANNEL} was not restored to {_PRE_RUN_MA} mA after the '
            f'abort: {aborted_composite["lit"]}'
        )

    def test_the_acquiring_channels_are_dark_after_the_abort(self, aborted_composite):
        # The other half of return-to-original: restoring the pre-run
        # channel while leaving the step's channel burning lights the
        # sample with a channel the user never turned on.
        still_lit = set(aborted_composite['lit']) & set(_ABORT_ACQUIRING)
        assert not still_lit, (
            f'the abort left acquiring channels lit: {sorted(still_lit)} '
            f'(board: {aborted_composite["lit"]})'
        )


# ---------------------------------------------------------------------------
# A run that finishes normally
# ---------------------------------------------------------------------------


class TestPreRunStateSurvivesACompleteRun:
    """The same hand-back on the path a user actually takes every time."""

    def test_a_pre_run_channel_is_lit_again_after_a_complete_run(self, tmp_path):
        # The abort case above exercises the restore through cleanup's
        # abort branch; this is the branch every successful composite
        # takes, and the two derive their end-state separately enough
        # that one can regress while the other stays green.
        settings = headless_settings(tmp_path, acquiring=('BF', 'Blue'))
        with open_composite_session(settings) as (session, runner):
            _light_pre_run_channel(session)
            assert _lit_channels(session) == {_PRE_RUN_CHANNEL: _PRE_RUN_MA}, (
                'the pre-run setup did not light the channel under test'
            )

            artifact = runner.run_composite(sequence_name='restore', parent_dir=str(tmp_path))
            assert pathlib.Path(artifact).exists(), 'the run under test did not complete'

            lit = _lit_channels(session)

        assert lit.get(_PRE_RUN_CHANNEL) == _PRE_RUN_MA, (
            f'{_PRE_RUN_CHANNEL} was not restored to {_PRE_RUN_MA} mA after a '
            f'completed composite: {lit}'
        )

    def test_the_acquiring_channels_are_dark_after_a_complete_run(self, tmp_path):
        # A composite that ends with its last step's channel still lit
        # bleaches the sample for as long as the user leaves the scope.
        settings = headless_settings(tmp_path, acquiring=('BF', 'Blue'))
        with open_composite_session(settings) as (session, runner):
            _light_pre_run_channel(session)
            runner.run_composite(sequence_name='restore', parent_dir=str(tmp_path))
            lit = _lit_channels(session)

        assert set(lit) == {_PRE_RUN_CHANNEL}, (
            f'the completed composite left channels lit beyond the pre-run state: {lit}'
        )


# ---------------------------------------------------------------------------
# Which channels light, and in what order
# ---------------------------------------------------------------------------


_ORDER_ACQUIRING = ('BF', 'PC', 'DF', 'Blue', 'Green')


@pytest.fixture
def ordered_composite(tmp_path, monkeypatch):
    """A composite with every transmitted channel set to acquire.

    Three transmitted channels compete for the one slot the merge has for
    them, so this is the shape where a capture order derived from
    anything but the catalogue shows up.
    """
    settings = headless_settings(tmp_path, acquiring=_ORDER_ACQUIRING)
    expected = get_composite_channels(settings)
    with open_composite_session(settings) as (session, runner):
        events = _record_led_commands(monkeypatch, session)
        artifact = runner.run_composite(sequence_name='order', parent_dir=str(tmp_path))
        yield {
            'expected': expected,
            'events': events,
            'artifact': artifact,
            'run_dir': single_run_dir(tmp_path),
        }


class TestCaptureOrder:
    """One transmitted channel maximum, lit in catalogue order."""

    def test_exactly_one_transmitted_channel_is_lit(self, ordered_composite):
        # All three transmitted channels occupy the same slot in the
        # merged image, so a second one is not an extra layer -- it is a
        # channel that overwrites the first, plus an exposure of the
        # sample that bought nothing.
        lit = _lit_sequence(ordered_composite['events'])
        transmitted = [color for color in lit if color in ('BF', 'PC', 'DF')]
        assert transmitted == ['BF'], (
            f'expected only BF of the transmitted channels to light, got {transmitted} '
            f'(full lit order {lit})'
        )

    def test_the_lit_order_is_the_catalogue_order(self, ordered_composite):
        # get_composite_channels is what the config assembly promises the
        # run will capture; a run that lights a different set, or the
        # same set in a different order, means the plan and the engine
        # disagree about what a composite IS.
        assert _lit_sequence(ordered_composite['events']) == ordered_composite['expected']

    def test_the_frames_on_disk_match_the_channels_that_lit(self, ordered_composite):
        # The merge reads frames off disk, so a channel that lit without
        # leaving a frame -- or a frame from a channel that never lit --
        # is a composite built from something other than what was
        # illuminated.
        expected = ordered_composite['expected']
        run_dir = ordered_composite['run_dir']
        assert _frame_channels(run_dir, _ORDER_ACQUIRING) == set(expected), (
            f'frames on disk do not match the captured channel set {expected}: '
            f'{sorted(p.name for p in run_dir.glob("*.tiff"))}'
        )
        assert len(list(run_dir.glob('*.tiff'))) == len(expected), (
            'the run wrote more frames than it captured channels'
        )


# ---------------------------------------------------------------------------
# The LED sequence within the run
# ---------------------------------------------------------------------------


_SEQUENCE_ACQUIRING = ('BF', 'Blue', 'Green', 'Lumi')

# Luminescence is emitted BY the sample; illuminating it is the one thing
# that destroys the measurement. It is also the only acquiring channel
# with no LED behind it, so it is the channel a run must capture DARK.
_LUMINESCENCE = 'Lumi'


@pytest.fixture
def sequenced_composite(tmp_path, monkeypatch):
    """A composite whose LED writes and frame grabs share one timeline."""
    settings = headless_settings(tmp_path, acquiring=_SEQUENCE_ACQUIRING)
    expected = get_composite_channels(settings)
    with open_composite_session(settings) as (session, runner):
        events = _record_led_commands(monkeypatch, session)
        _record_captures(monkeypatch, events)
        runner.run_composite(sequence_name='sequence', parent_dir=str(tmp_path))
        yield {
            'expected': expected,
            'events': events,
            'run_dir': single_run_dir(tmp_path),
        }


def _replay(events):
    """Walk the recorded timeline, yielding (event, lit-set) after each."""
    lit = set()
    for kind, color, illumination_ma in events:
        if kind == 'on':
            lit.add(color)
        elif kind == 'off':
            lit.discard(color)
        elif kind == 'all_off':
            lit.clear()
        yield (kind, color, illumination_ma), set(lit)


class TestOneChannelAtATime:
    """The run drives exactly one channel, and puts it out afterwards."""

    def test_two_channels_are_never_lit_together(self, sequenced_composite):
        # Every per-channel frame must be exposed by ITS channel alone. A
        # boundary that held one channel while the next lit would bleed
        # the previous colour into the frame, and the merged composite
        # would look plausible with the wrong colours in it.
        for event, lit in _replay(sequenced_composite['events']):
            assert len(lit) <= 1, f'{sorted(lit)} were lit together, after {event}'

    def test_each_channel_goes_dark_after_its_own_capture(self, sequenced_composite):
        # The capture marker sits between the step's illuminate and its
        # boundary decision, so this is the ordering that says the sample
        # is lit only for as long as the grab needs.
        events = sequenced_composite['events']
        captured = [color for kind, color, _ in events if kind == 'capture']
        assert captured == sequenced_composite['expected'], (
            f'the run captured {captured}, not the planned channel set '
            f'{sequenced_composite["expected"]}'
        )
        for channel in ('BF', 'Blue', 'Green'):
            timeline = [
                (kind, color) for kind, color, _ in events if color == channel or kind == 'all_off'
            ]
            assert ('capture', channel) in timeline, f'{channel} was never captured'
            grab = timeline.index(('capture', channel))
            assert ('on', channel) in timeline[:grab], (
                f'{channel} was captured without being lit first: {timeline}'
            )
            assert any(kind in ('off', 'all_off') for kind, _ in timeline[grab + 1 :]), (
                f'{channel} was never extinguished after its capture: {timeline}'
            )

    def test_luminescence_is_never_lit(self, sequenced_composite):
        # A luminescence channel measures light the sample emits; driving
        # an LED during that step does not merely add background, it
        # swamps the signal the step exists to record.
        for kind, color, illumination_ma in sequenced_composite['events']:
            assert not (kind == 'on' and color == _LUMINESCENCE), (
                f'the run drove the luminescence channel on at {illumination_ma} mA'
            )

    def test_luminescence_still_contributes_a_frame(self, sequenced_composite):
        # Not lighting it is not the same as skipping it: the merge needs
        # the dark-field emission frame, and a step silently dropped for
        # having no LED would leave the composite a channel short.
        assert _LUMINESCENCE in sequenced_composite['expected'], (
            'the config assembly dropped the luminescence channel'
        )
        assert _frame_channels(sequenced_composite['run_dir'], (_LUMINESCENCE,)) == {
            _LUMINESCENCE
        }, (
            'the luminescence step wrote no frame: '
            f'{sorted(p.name for p in sequenced_composite["run_dir"].glob("*.tiff"))}'
        )
