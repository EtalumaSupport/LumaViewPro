# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope doubles that cannot answer for attributes the real scope lacks.

A bare `MagicMock()` scope says yes to everything: `hasattr` is True for
every name, every attribute access invents a child mock, and every call
returns one. A test written against it passes whether or not the code
under test asks the real `Lumascope` for something it has. Two families
of production-dead code stayed green in this suite for exactly that
reason (see `tests/guards/test_capability_probe_reality.py`).

`spec_scope()` builds its double by autospec'ing a CONSTRUCTED
`Lumascope(simulate=True)` INSTANCE. Accessing a name the real scope
does not have raises `AttributeError` instead of inventing a child, and
calling a method with the wrong signature raises `TypeError`.

Autospec the INSTANCE, never the class. The sub-APIs
(`illumination`, `imaging`, `motion`, `diagnostics`, `protocols`,
`runtime_state`) and the driver slots are assigned in
`Lumascope.__init__`. A class autospec therefore has none of them, so it
would reject `scope.illumination.led_on(...)` -- legitimate production
access -- while the instance autospec accepts it and still rejects what
the real object genuinely lacks. Getting this backwards inverts the
guard into an obstacle, which is why it is stated here rather than left
to be rediscovered.

Which double to reach for
-------------------------

Prefer the leftmost option that works; each step right buys isolation
by giving up realism.

1. **`sim_scope` (in `conftest.py`) is the DEFAULT for module-layer
   tests.** A real `Lumascope` on simulated drivers: real sub-APIs, real
   state transitions, real signatures. If the code under test just needs
   a working scope, use this and nothing here.

2. **`spec_scope()` (this module) when `sim_scope` is too heavy or when
   the test must inject failure.** A spec'd double costs no driver
   construction and lets you set `side_effect` to raise, which a real
   simulated scope will not do on demand. You give up real behavior:
   nothing downstream of the double actually happens.

3. **`camera_fakes` for driver-BEHAVIOR tests.** Real driver objects
   with a fake SDK underneath. A different layer, not a competing
   choice -- it answers "what does the driver do", where this module
   answers "what does the caller do with a scope".

A bare `MagicMock()` scope is not on this list. The ratchet in
`test_scope_fakes.py` records how many test files still build one and
does not let that number grow.
"""

from __future__ import annotations

import copy
import time
import weakref
from unittest.mock import create_autospec


#: What `build_scope` and `give_stub_lanes` built and no test teardown has
#: stopped yet, oldest first, as the call that stops each. The autouse fixture
#: in `conftest.py` runs the ones a test built when it ends; a scope a
#: module-scoped fixture built before the test began is that fixture's to
#: disconnect.
_TEARDOWNS: list = []


def build_scope(**kwargs):
    """A `Lumascope` for a test: the one place a test constructs one.

    Takes the constructor's own arguments. A test that needs a scope of
    its own, rather than a session's, builds it here, so what every such
    scope is built with changes in one place.

    The test's teardown disconnects it, which stops the threads it runs;
    a test may disconnect it earlier itself, since disconnect repeats
    safely.

    A simulated scope's camera takes the shipped template's frame, as
    bring-up applies the stored one on every host. A bare scope is never
    brought up, so its camera would otherwise stream its whole sensor, a
    size the instrument does not run at and several times the frame to
    render.

    A real scope a test names no folder for is built on an installation of
    its own (``installation_of_its_own``): it takes its installation's
    instance lock, which on the repo's folder would meet a LumaViewPro
    running beside the suite.
    """
    from modules.lumascope_api import Lumascope
    from tests.settings_fixtures import complete_settings, installation_of_its_own

    if not kwargs.get('simulate') and kwargs.get('source_path') is None:
        kwargs['source_path'] = installation_of_its_own()
    # A simulated scope no test names a model for is the LS850T, the
    # model every bare test scope has been built on; a scope with no
    # session's settings would otherwise take the template's model, and
    # the tests that home or assign a turret would lose it.
    if kwargs.get('simulate') and not (kwargs.get('sim_model') or kwargs.get('configured_model')):
        kwargs['sim_model'] = 'LS850T'
    scope = Lumascope(**kwargs)
    _TEARDOWNS.append(scope.disconnect)
    if kwargs.get('simulate') and scope.camera_connected:
        frame = complete_settings()['frame']
        scope.camera._set_frame_size_impl(frame['width'], frame['height'])
    return scope


def disconnect_when_the_test_ends(camera):
    """Queue a simulated camera's disconnect with the test's other teardowns.

    A simulated camera that is grabbing runs its own acquisition thread, as
    a real camera's SDK thread runs, until it is disconnected. Tests build
    them directly in many places and drop them unclosed; each one left
    behind keeps making frames for the rest of the worker's run. The
    autouse fixture in `conftest.py` routes every construction here.
    """
    _TEARDOWNS.append(camera.disconnect)


# How long a session's teardown lets work a test left live end on its own
# before it fails the test rather than wait on a close that cannot return.
_LEFT_WORK_WAIT_S = 30.0


def real_executor_bundle(**members):
    """A started executor bundle of real lanes around two fresh device lanes.

    A session's close asks its lanes what they are still doing and waits
    until nothing is; a real idle lane answers that by being idle, where a
    MagicMock answers every such question with a truthy mock and the close
    never ends. *members* replace bundle members a test injects (a file lane
    it scripts); everything the bundle started is stopped when the test ends.
    """
    import dataclasses

    from modules.executor_registry import create_default
    from modules.sequential_io_executor import SequentialIOExecutor

    io, camera = SequentialIOExecutor(name='IO'), SequentialIOExecutor(name='CAMERA')
    io.start()
    camera.start()
    bundle = create_default(io, camera)

    def stop() -> None:
        bundle.shutdown()
        io.shutdown(wait=False)
        camera.shutdown(wait=False)

    _TEARDOWNS.append(stop)
    return dataclasses.replace(bundle, **members) if members else bundle


def shut_down_when_the_test_ends(session):
    """Queue a session's shutdown with the test's other teardowns.

    A session runs periodic timers on its own scheduler (the stream check
    among them) until it is shut down, and each tick re-arms on a new timer
    thread. Tests build sessions in many places and drop them unclosed; one
    left behind keeps starting threads for the rest of the worker's run,
    inside the window of every later test that checks a refused bring-up
    started nothing. The autouse fixture in `conftest.py` routes every
    construction here; a second shutdown() is a logged no-op.

    The close waits, without a time limit, for the work the session is
    still doing. A test that leaves work live -- a claim it took and never
    released, a run's writes it planted and never settled -- would hang the
    worker there, so that work is given a short while to end, and if it has
    not, the test fails naming it and the close is not called.
    """

    def stop() -> None:
        deadline = time.monotonic() + _LEFT_WORK_WAIT_S
        while session.live_work.work and time.monotonic() < deadline:
            time.sleep(0.02)
        left = session.live_work.work
        if left:
            raise AssertionError(f'the test left work live, so its session cannot close: {left}')
        session.shutdown()

    _TEARDOWNS.append(stop)


def give_stub_lanes(scope):
    """Give a `Lumascope.__new__` stub the two started lanes a built scope has.

    A stub skips the constructor, so it has no lanes, and every dispatcher
    runs its command on one; nor has it the claim slot beside them, empty as
    on a scope no session composes, which every guarded write reads. The
    test's teardown shuts the lanes. Returns the scope.
    """
    from modules.sequential_io_executor import SequentialIOExecutor

    scope._io_executor = SequentialIOExecutor(name='IO')
    scope._camera_executor = SequentialIOExecutor(name='CAMERA')
    scope._camera_override_key = None
    scope._activity_claim = None
    for lane in (scope._io_executor, scope._camera_executor):
        lane.start()
        _TEARDOWNS.append(lane.shutdown)
    return scope


def give_camera_capabilities(scope, camera):
    """Give a `Lumascope.__new__` stub the capabilities a built scope reads off `camera`.

    Built by the production constructor with no boards and no model, so the
    stub's static camera facts are the ones its camera reports. Returns the
    scope.
    """
    from drivers.null_ledboard import NullLEDBoard
    from drivers.null_motorboard import NullMotionBoard
    from modules.layer_record import UNRESOLVED
    from modules.scope_capabilities import ScopeCapabilities

    scope.capabilities = ScopeCapabilities.from_drivers(
        motion=NullMotionBoard(),
        led=NullLEDBoard(),
        camera=camera,
        channel_identity=UNRESOLVED,
        scope_models={},
        timing_record=None,
    )
    return scope


def give_capabilities_of_a_camera_without_auto_modes(scope):
    """Give a stub over a duck-typed fake driver the capabilities its camera and capture read.

    No ``ScopeCapabilities`` can be built from such a fake (it reports no
    model or profile), and a built scope always has them before its camera
    API, which reads whether the camera has hardware auto-gain and
    auto-exposure as it brings its held modes up. This camera has neither,
    so the API holds both Off and never asks the fake. A capture reads the
    active objective, which asks whether the scope has a turret (this one
    has none) and records its optics (unknown here). Returns the scope.
    """
    import types

    # a stand-in by design: the fake reports nothing a ScopeCapabilities is
    # built from, and the tests using it are about other members.
    scope.capabilities = types.SimpleNamespace(
        camera_supports_auto_gain=False,
        camera_supports_auto_exposure=False,
        has_turret=False,
        pixel_size_um=None,
        lens_focal_length_mm=None,
    )
    return scope


def give_camera_and_imaging(scope):
    """Give a `Lumascope.__new__` stub the camera and capture sub-APIs a built scope has.

    Built as the constructor builds them: the scope's one frame validity,
    then ``scope.camera`` over it, then ``scope.imaging`` with both. The stub
    needs its camera driver (``_camera_driver``), lanes and capabilities set
    first, since the camera reads its cache from the driver as it is built.
    It is given the objective catalogue too, which a capture reads for the
    active objective's scale. Returns the camera.
    """
    from modules.frame_validity import FrameValidity
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.camera import CameraAPI
    from modules.lumascope_api.imaging import ImagingAPI
    from modules.objectives_loader import ObjectiveLoader

    scope.objective_helper = ObjectiveLoader()
    scope.frame_validity = FrameValidity(Lumascope._frames_delivered.__get__(scope))
    scope.camera = CameraAPI(scope, scope.frame_validity)
    scope.imaging = ImagingAPI(scope, scope.camera, scope.frame_validity)
    return scope.camera


def set_camera_limits(monkeypatch, scope, **limits):
    """Give ``scope`` a camera whose limits differ from its camera's by ``limits``.

    ``capabilities.camera_limits`` is read once at connect, so a test that
    needs a camera declaring another range (a floor above 0 dB, no floor at
    all) states it here; the test's end restores the scope's own. Returns
    the new limits.
    """
    import dataclasses

    changed = dataclasses.replace(scope.capabilities.camera_limits, **limits)
    monkeypatch.setattr(
        scope, 'capabilities', dataclasses.replace(scope.capabilities, camera_limits=changed)
    )
    return changed


def swap_lanes(scope, *, io=None, camera=None):
    """Put a test's own lane in place of a scope's: the one sanctioned lane seam.

    A scope builds its own IO and CAMERA lanes, and the session, the run
    engine and cleanup all read them from it, so a test that needs a
    recording, stalled or double lane puts it here and every reader sees
    it. On a real scope the lane it replaces is shut; on a mock scope the
    lane is what ``io_lane()`` / ``camera_lane()`` answer. The test owns
    the lane it passed. Returns the scope.
    """
    from unittest.mock import Mock

    for lane, getter, slot in (
        (io, 'io_lane', '_io_executor'),
        (camera, 'camera_lane', '_camera_executor'),
    ):
        if lane is None:
            continue
        if isinstance(scope, Mock):
            getattr(scope, getter).return_value = lane
        else:
            getattr(scope, slot).shutdown(wait=False)
            setattr(scope, slot, lane)
    return scope


def teardown_mark() -> int:
    """How many teardowns are waiting: the mark a test starts at."""
    return len(_TEARDOWNS)


def tear_down_since(mark: int) -> None:
    """Stop everything `build_scope` and `give_stub_lanes` built after ``mark``."""
    built = _TEARDOWNS[mark:]
    del _TEARDOWNS[mark:]
    failures = []
    for stop in built:
        try:
            stop()
        except Exception as failed:
            failures.append(failed)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup("the test's teardowns failed", failures)


def build_real_sim_scope():
    """A constructed `Lumascope(simulate=True)`, for use as the spec.

    Production defaults are kept (`register_atexit` left on) so the spec
    covers everything a production caller can reach.

    The test's teardown disconnects it; `spec_scope()` disconnects its own
    at once. Bound to the template's settings: the spec reads every
    attribute, and the scale bar's answer reads the settings.
    """
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    return scope


def homed_sim_scope():
    """A `Lumascope(simulate=True)` that has been homed, ready to move.

    Axes start UNKNOWN, and the motion gate refuses to drive an axis
    whose position is unknown -- so a test that commands a move without
    homing first is asking for something production never does. The App
    homes at startup before any protocol or jog is reachable; this is
    that precondition, for the tests whose subject is something else
    (LED ordering, protocol flow, frame validity) and for which motion
    is only a fixture.

    Homing runs the production body against the real simulated driver.
    Only the simulator's artificial 3-second homing sleep is skipped --
    that models how long a real stage takes to travel, not what any of
    this does, and paying it once per test would cost minutes of suite
    time. The scope's timing mode is left exactly as constructed.

    The test's teardown disconnects it.
    """
    return home_sim_scope(build_real_sim_scope())


def home_sim_scope(scope):
    """Home an already-built simulated scope, and return it.

    The same precondition as `homed_sim_scope`, for the fixtures that
    build and configure their scope themselves. Restores the timing mode
    it found, so a fixture that set one keeps it.
    """
    driver = scope._motion_driver
    prior_timing = driver._timing_mode
    driver.set_timing_mode('instant')
    try:
        # A home that fails raises HomingFailedError, naming why.
        scope.motion._home_impl()
    finally:
        driver.set_timing_mode(prior_timing)
    return scope


def led_reading(scope, address):
    """The reading of one LED board address, named or not, from ``get_led_states()``.

    ``get_led_state`` takes a channel's name; a test about an address, or
    about one the unit's identity does not name, reads it here.
    """
    (reading,) = [r for r in scope.illumination.get_led_states() if r.address == address]
    return reading


def spec_scope(**attrs):
    """A scope double specced against a real constructed Lumascope.

    Args:
        **attrs: attributes to set on the double after construction, for
            the values the test actually cares about. Names not present
            on the real scope are rejected here rather than silently
            accepted, so a typo in a test fails loudly.

    Returns:
        A `MagicMock` specced to the real scope: unknown attribute ->
        `AttributeError`, wrong call signature -> `TypeError`, sub-APIs
        present and specced to their own instances.

    Example:
        scope = spec_scope(camera_connected=True)
        scope.illumination.led_on(channel='BF', illumination_ma=100)   # ok
        scope.led_on_fast(channel='BF', illumination_ma=100)           # AttributeError
    """
    real = build_real_sim_scope()
    try:
        # a stand-in by design: the specced double every spec_scope caller takes
        double = create_autospec(real, instance=True, spec_set=True)
        # A session's factory prepares the settings it is handed against
        # the scope's own template, read from the scope's root: the
        # double answers both as the real scope does, not as a mock.
        double.settings_template = real.settings_template
        double.source_path = real.source_path
        # Each sub-API's spec is taken on first access, from the real one:
        # taken here, while its drivers are up, since a read the API refuses
        # with no camera (imaging.significant_bits) would raise from a
        # disconnected scope's.
        for sub_api in (
            'camera',
            'capabilities',
            'diagnostics',
            'illumination',
            'imaging',
            'motion',
            'protocols',
            'runtime_state',
        ):
            getattr(double, sub_api)
    finally:
        # The spec is captured by create_autospec; the live object has
        # served its purpose and must not keep simulated drivers open.
        real.disconnect()

    for name, value in attrs.items():
        # spec_set makes this raise AttributeError for a name the real
        # scope lacks -- the point of the fixture, so it is not caught.
        setattr(double, name, value)
    return double


# The objectives a test turret carries. Every slot but the last is filled,
# because the protocols across this suite name three different objectives
# and a scope that carries all of them is one fewer thing for a test about
# something else to have to configure.
TEST_TURRET_OBJECTIVES = {1: '10x Oly', 2: '20x Oly', 3: '4x Oly', 4: None}


# The settings each scope bound here reads, so a second call adjusts them:
# a scope is bound once (``Lumascope.bind_settings`` refuses a second), and a
# fixture that bound the template and a test that names its own plate both
# describe the same session.
_BOUND_SETTINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def bind_settings_like_a_session(scope, **overrides) -> dict:
    """Give a hand-built test scope the settings a session would bind it to.

    The scope reads its labware, stage offset, turret map, objective and
    scale bar from a session's settings, and its camera saves the settings it
    changes there (``Lumascope.bind_settings``); a scope no session bound
    refuses both. A test about something else binds its bare scope here, to
    the shipped template with ``overrides`` laid over it
    (``complete_settings``), read and written through the Session's own
    reader and writer so the answers are the ones a session gives. Called again for a
    scope it bound, it lays ``overrides`` over the settings already bound.

    Returns:
        The settings dict the scope now reads; a test may change it.
    """
    import functools
    import types

    from modules.activity_claim import ActivityClaim
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    if scope in _BOUND_SETTINGS:
        settings = _BOUND_SETTINGS[scope]
        settings.update(complete_settings(**{**settings, **copy.deepcopy(overrides)}))
        return settings
    # A `Lumascope.__new__` stub skipped the constructor that starts it unbound.
    vars(scope).setdefault('_settings_reader', None)
    assert scope._settings_reader is None, (
        "the scope already reads a session's settings: change them through the session"
    )
    settings = complete_settings(**overrides)
    _BOUND_SETTINGS[scope] = settings
    from modules.scope_session import SettingsLock

    holder = types.SimpleNamespace(
        settings=settings,
        # The store's own lock, which records what _store_setting stores; told to no one.
        settings_lock=SettingsLock(lambda paths: None),
        _settings_saver=None,
        activity_claim=ActivityClaim(),
    )
    holder._store_setting = functools.partial(ScopeSession._store_setting, holder)
    scope.bind_settings(
        functools.partial(ScopeSession._read_stored_setting, holder),
        functools.partial(ScopeSession._store_settings, holder),
    )
    return settings


def configure_turret_like_bringup(scope, turret_objectives: dict | None = None) -> None:
    """Give a hand-built test scope the turret state a real one comes up with.

    `Lumascope(simulate=True)` is not a scope that has been brought up. It
    reports the only turreted model as its own (the simulate branch falls
    back to LS850T when settings are not yet loaded), while its runtime
    turret configuration stays empty -- a turret whose every slot is
    clear. A real session never looks like that: the scope reads the
    persisted slots from its session's settings, and on a turreted scope the
    startup objective question assigns the current position before any
    protocol can be loaded.

    An empty turret addresses no glass at all, so without this every
    protocol is refused, whatever the test was about. Test scopes that
    run protocols call this for the same reason the ones here already
    register a source path by hand: it is a step of bring-up that a bare
    scope skipped, not a fact about the test.

    Args:
        scope: The Lumascope to configure.
        turret_objectives: Slot -> objective id, defaulting to
            TEST_TURRET_OBJECTIVES. Pass a narrower one to test a scope
            that genuinely carries less.
    """
    # The stage offset is the one a run's travel check and its plate
    # conversions read.
    bind_settings_like_a_session(
        scope,
        turret_objectives=dict(
            TEST_TURRET_OBJECTIVES if turret_objectives is None else turret_objectives
        ),
        stage_offset={'x': 0.0, 'y': 0.0},
    )
    # Startup homes every axis: the turret lands in slot 1 -- the active objective is that
    # slot's assignment, unknown until the slot is -- and a run's moves need
    # a known stage, since the run moves every step itself.
    home_sim_scope(scope)


def answer_auto_gain_like_the_api(camera, *, has_auto_gain: bool = True) -> None:
    """Give a fake camera surface the API's own ``applied_auto_gain_for``.

    A bare MagicMock answers a MagicMock, which is truthy, so a caller asking
    whether a step's auto-gain applies would always hear yes. This wires the
    real member over a camera that does or does not have hardware auto-gain,
    so the fake gives the API's decision rather than one of its own.
    """
    from modules.lumascope_api.camera import CameraAPI

    camera._camera_has_auto_gain = lambda: has_auto_gain
    camera.applied_auto_gain_for = CameraAPI.applied_auto_gain_for.__get__(camera)


def identity_with_layers(keys, *, model=None, filterset=''):
    """A unit carrying the layers ``keys``, resolved by the real resolver.

    Each layer's row is the one the shipped catalogue's models state for it
    (the first model listing the key), and the rows are resolved as a unit's
    own block is, against the shipped release catalogue: every record
    carries the fields the production resolver fills.
    """
    from modules import layer_record

    shipped = {}
    for entry in layer_record.load_scope_models().values():
        for record in entry.channels:
            shipped.setdefault(record.key_name, record)
    channels = {
        key: {
            'display_name': shipped[key].display_name,
            'led_channel': list(shipped[key].led_addresses) or None,
            'excitation_nm': shipped[key].excitation_nm,
        }
        for key in keys
    }
    return layer_record.resolve_layer_identity(
        board_block={'Channels': channels, 'Filterset': filterset},
        board_config_read_ok=True,
        motor_model=model,
        configured_model=None,
        models={},
        catalogue=layer_record.release_layers(),
        report=refuse_a_report,
    )


def stock_unit():
    """A stock LS850's identity, resolved from the release's model rows.

    The unit a test reads a protocol to run on when the test is not about
    the unit: each channel name the file holds is read as this unit's key
    for it (``Protocol.from_file``'s ``unit``).
    """
    from modules import layer_record

    return layer_record.resolve_layer_identity(
        board_block=None,
        board_config_read_ok=True,
        motor_model='LS850',
        configured_model=None,
        models=layer_record.load_scope_models(),
        catalogue=layer_record.release_layers(),
        report=refuse_a_report,
    )


def refuse_a_report(notice):
    """An identity reporter for a resolve that has nothing to report."""
    raise AssertionError(f'the identity resolve reported {notice!r}')
