# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture asked for after a change returns a frame exposed after the change was in force.

Frame validity places a change in the host's arrival order and credits any
frame that arrives after it, so a frame exposed before the change but still
on its way to the host -- in the link, in the driver's queue -- is credited
and can be returned as the capture. On the simulator's realistic tier, with
live view crediting frames and the host held a fixed backlog behind the
camera, each stale-frame member's change is followed by ``capture_still``
and the frame it returned is judged by what the simulated sensor made it
under (``scope.simulation.exposed_under``): never by the gate's own answer,
never by pixels.

Counted frames decide: the simulated sensor shows a change in the exposures
starting after it, its pipeline still delivering the frame it had started,
and each source waits its shipped count, the simulator's timing record
measuring none, as the shipped cameras' measure none; a move's count runs
from its stop. A turret move parks and restores Z inside its span, so its
frame waits out the same stop. Each change of a camera setting, the LED and
the stage is taken at a short exposure and a long one too, since how many
frames are in flight at a change depends on the exposure. Conversion gain mode and line noise reduction have no case: the simulated
camera is the dart's, which has neither, and they are not simulated (Eric,
2026-10-11).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from _lumascope.modules.exceptions import CaptureFailedError
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import TEST_TURRET_OBJECTIVES
from tests.settings_fixtures import complete_settings

# Frames held between the camera and the host, more than any source's count
# (3, exposure's): a capture that counted frames as they reached the host,
# rather than as the driver was handed them, would take one made before the
# change on every run.
BACKLOG = 6
# The frames the listener keeps, so the one a capture names is still held.
FRAMES_KEPT = 64
LIVE_VIEW_PERIOD_S = 1 / 30


@dataclass(frozen=True)
class Case:
    """One stale-frame member: what is set before, the change, and whether a
    frame's exposure record shows it was made after the change.

    ``change`` makes the change and returns what the scope reads once it has
    landed; ``made_after(exposure, landed)`` compares the record with it.
    """

    member: str
    before: Callable
    change: Callable
    made_after: Callable


def _same(a: float, b: float) -> bool:
    return a == pytest.approx(b, rel=1e-9)


def _z_landed(scope):
    return scope.motion.get_current_position('Z')


def _z_steady_at(exposure, z):
    return exposure.z_um == z and exposure.z_end_um == z


def _xy_move(scope, x, y):
    scope.motion.move_xy(x_um=x)
    scope.motion.move_xy(y_um=y)
    return scope.motion.get_current_position('X'), scope.motion.get_current_position('Y')


def _turret_move(scope, slot):
    scope.motion.move_turret(slot)
    dof = float(scope.objectives[TEST_TURRET_OBJECTIVES[slot]].depth_of_field_um)
    return dof, _z_landed(scope)


CASES = (
    Case(
        'exposure',
        lambda s: s.camera.set_exposure_ms(10.0),
        lambda s: (s.camera.set_exposure_ms(20.0), s.camera.read_exposure_ms())[1],
        lambda e, ms: _same(e.exposure_us, ms * 1000.0),
    ),
    Case(
        'gain',
        lambda s: s.camera.set_gain_db(0.0),
        lambda s: (s.camera.set_gain_db(6.0), s.camera.read_gain_db())[1],
        lambda e, db: _same(e.gain_db, db),
    ),
    Case(
        'black_level',
        lambda s: s.camera.set_black_level(2.0),
        lambda s: (s.camera.set_black_level(4.0), s.camera.read_black_level())[1],
        lambda e, level: _same(e.black_level, level),
    ),
    # From one lit current to another: a dark frame would be refused by the
    # capture's dark-floor retry, which is not the gate under test.
    Case(
        'led',
        lambda s: s.illumination.led_on('BF', 50.0),
        lambda s: s.illumination.led_on('BF', 100.0),
        lambda e, ma: _same(e.illumination_ma, ma),
    ),
    Case(
        'z_move',
        lambda s: s.motion.move_z(5_000.0),
        lambda s: (s.motion.move_z(5_200.0), _z_landed(s))[1],
        _z_steady_at,
    ),
    Case(
        'xy_move',
        lambda s: _xy_move(s, 20_000.0, 20_000.0),
        lambda s: _xy_move(s, 21_000.0, 21_000.0),
        lambda e, xy: (e.x_um, e.y_um) == xy,
    ),
    # The objective is unknown while the turret turns, so a frame exposed
    # then has no depth of field, and one exposed before has the 10x's.
    Case(
        'turret',
        lambda s: _turret_move(s, 1),
        lambda s: _turret_move(s, 2),
        lambda e, landed: e.depth_of_field_um == landed[0] and _z_steady_at(e, landed[1]),
    ),
)


@pytest.fixture(scope='module')
def realistic_session(tmp_path_factory):
    """A homed LS850T session on the realistic tier, the camera and the stage
    taking a real scope's time, with the test objectives on its turret."""
    session = ScopeSession.create(
        complete_settings(
            microscope='LS850T',
            simulator_tier='realistic',
            turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            live_folder=str(tmp_path_factory.mktemp('stale_frame')),
        ),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        session.scope.motion.home()
        yield session
    finally:
        session.shutdown()


@pytest.fixture
def live_view(realistic_session):
    """Live view running, as the app's preview grabs from the buffer and
    credits the frames it shows."""
    imaging = realistic_session.scope.imaging
    stop = threading.Event()

    def show():
        while not stop.wait(LIVE_VIEW_PERIOD_S):
            imaging.buffered_image()

    viewer = threading.Thread(target=show, name='live view', daemon=True)
    viewer.start()
    yield
    stop.set()
    viewer.join()


class Delivered:
    """The frames the camera delivers, by their ``FrameID``, and a wait for the next."""

    def __init__(self):
        self._frames = {}
        self._count = 0
        self._arrived = threading.Condition()

    def keep(self, frame):
        with self._arrived:
            self._frames[frame.chunks['FrameID']] = frame
            self._count += 1
            while len(self._frames) > FRAMES_KEPT:
                del self._frames[next(iter(self._frames))]
            self._arrived.notify_all()

    def __getitem__(self, frame_id):
        with self._arrived:
            return self._frames[frame_id]

    def next_one(self):
        """Wait for the next frame to arrive; a backlog asked for stands once one does.

        Raises TimeoutError, not an assertion, so a camera that stopped
        delivering fails as itself, never as a stale frame.
        """
        with self._arrived:
            seen = self._count
            if not self._arrived.wait_for(lambda: self._count > seen, timeout=5.0):
                raise TimeoutError('no frame arrived')


@pytest.fixture
def delivered(realistic_session):
    """The frames the camera delivers from now on."""
    frames = Delivered()
    imaging = realistic_session.scope.imaging
    imaging.add_frame_listener(frames.keep, name='frames delivered')
    yield frames
    imaging.remove_frame_listener(frames.keep)


def _capture_after(scope, delivered, case):
    """The change of ``case`` made behind a host backlog, a capture after it,
    and what the frame it returned was made under, with what the change landed."""
    case.before(scope)
    scope.imaging.capture_still()
    scope.simulation.delay_delivery(BACKLOG)
    try:
        delivered.next_one()
        landed = case.change(scope)
        try:
            record = scope.imaging.capture_still().record
        except CaptureFailedError as failed:
            # No frame made after the change within the capture's bound:
            # the contract's other side, a capture that takes the next
            # frame, missed. Any other failure is not the one this test
            # expects.
            if failed.reason != 'no_fresh_frame':
                raise
            raise AssertionError(
                f'the capture gave up on {failed.source} ({failed.reason})'
            ) from failed
        made_under = scope.simulation.exposed_under(delivered[record.frame_id])
    finally:
        scope.simulation.clear()
    return made_under, landed


@pytest.mark.slow
# parametrize-justified: each member is its own change and its own session state.
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.member)
def test_a_capture_after_a_change_returns_a_frame_exposed_after_it(
    realistic_session, live_view, delivered, case
):
    made_under, landed = _capture_after(realistic_session.scope, delivered, case)

    assert case.made_after(made_under, landed), (landed, made_under)


# The camera's shortest exposure here and the LS850T's long one: one frame in
# flight at a change at the delivery ceiling, and one exposing for 134 ms.
EXPOSURES_MS = (1.0, 134.0)
AT_EVERY_EXPOSURE = ('gain', 'led', 'xy_move', 'z_move')


@pytest.mark.slow
# parametrize-justified: each change at each exposure is its own frame timing.
@pytest.mark.parametrize('exposure_ms', EXPOSURES_MS, ids=lambda ms: f'{ms:g}ms')
@pytest.mark.parametrize(
    'case',
    [case for case in CASES if case.member in AT_EVERY_EXPOSURE],
    ids=lambda case: case.member,
)
def test_a_capture_after_a_change_at_a_short_or_a_long_exposure_returns_a_frame_exposed_after_it(
    realistic_session, live_view, delivered, case, exposure_ms
):
    scope = realistic_session.scope
    scope.camera.set_exposure_ms(exposure_ms)
    try:
        made_under, landed = _capture_after(scope, delivered, case)
    finally:
        scope.camera.set_exposure_ms(10.0)

    assert case.made_after(made_under, landed), (landed, made_under)
