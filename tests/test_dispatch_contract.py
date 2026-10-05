# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The dispatch contract for the public hardware members.

Each public hardware member is a DISPATCHER over a private ``_impl`` that
holds the body. Every internal caller and both public tiers bind ``_impl``
directly, so the dispatcher is reached only by external callers -- an SDK
script, a REST handler, a MATLAB client -- who are never on an executor
worker and never on the protocol or autofocus thread.

Every scope builds its own IO and CAMERA lanes, a bare ``Lumascope()`` in
a script included, so there is no branch that runs the body on the calling
thread. Two branches, and this file pins them:

  * the lane will not accept work -> raise HardwareCommandRefusedError,
    saying which: another activity has the scope, or the lane was shut
    when the scope disconnected.
  * otherwise -> submit to the lane and block for the result.

The middle branch asks WHETHER the executor accepts work, never WHY it
might not. A run disables the camera executor while io and file are fenced
by protocol_start(); ``put()`` returns None for both states and the caller
cannot tell them apart. A branch keyed on "is a protocol running" lets a
camera write submit, receive None, and silently reach no hardware -- so
both states are driven here and both must refuse identically.

Parametrized across all three families because they do not share a dispatch
shape: each has its own dispatcher (``_dispatch_led``, ``_dispatch_camera``,
``_dispatch_motion``) with its own wait bound. A contract pinned on one
family would not catch the other two diverging from it. Motion is pinned on
its start member, whose body is the lane's whole work; the waited
``move_absolute`` is that start followed by a wait off the lane.

The probe replaces ``_impl`` with a recorder rather than asserting hardware
side effects: what is under test is WHERE and WHETHER the body runs, not
what the body does. The recorder's captured thread name is the evidence --
the calling thread for a direct call, the executor's worker for a submit.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from modules.exceptions import HardwareCommandRefusedError
from tests.scope_fakes import homed_sim_scope

# Sentinel returned by the probe. The dispatcher must hand it back on both
# non-refusing branches: a direct call returns what _impl returned, and a
# submit-and-block returns what the worker produced. A dispatcher that
# submits without waiting returns None here and fails.
IMPL_RESULT = object()

# (family, member, kwargs, which executor carries it)
#
# Every family's dispatcher is its only public form: the fire-and-forget
# tiers were deleted, so there is no second spelling of a command to pin.
FAMILIES = [
    ('illumination', 'led_on', {'channel': 0, 'illumination_ma': 10.0}, 'io'),
    ('imaging', 'set_gain_db', {'gain_db': 1.0}, 'camera'),
    (
        'motion',
        'start_move_absolute',
        {'axis': 'Z', 'position': 100.0},
        'io',
    ),
    # The camera-settings cluster: every public camera-state writer is a
    # dispatcher over its _impl, same three-branch contract. These were
    # the inline-invalidator family -- public bodies that wrote the
    # camera bus and frame validity on the caller's thread, unfenceable
    # by a running protocol -- and this table is what keeps them (and
    # any future sibling) on the dispatcher.
    (
        'imaging',
        'set_auto_gain',
        {
            'state': True,
            'settings': {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0},
        },
        'camera',
    ),
    ('imaging', 'set_auto_exposure_time', {'state': True}, 'camera'),
    ('imaging', 'set_frame_size', {'w': 640, 'h': 480}, 'camera'),
    ('imaging', 'set_binning_size', {'size': 1}, 'camera'),
    ('imaging', 'set_pixel_format', {'pixel_format': 'Mono8'}, 'camera'),
    ('imaging', 'set_conversion_gain_mode', {'mode': 'High'}, 'camera'),
    ('imaging', 'set_line_noise_reduction', {'enabled': True}, 'camera'),
    ('imaging', 'set_black_level', {'value': 4.0}, 'camera'),
    (
        'imaging',
        'update_auto_gain_target_brightness',
        {'target_brightness': 0.5},
        'camera',
    ),
    (
        'imaging',
        'auto_gain_once',
        {
            'state': True,
            'target_brightness': 0.3,
            'min_gain_db': 0.0,
            'max_gain_db': 20.0,
        },
        'camera',
    ),
    (
        'imaging',
        'apply_layer_camera_settings',
        {'gain_db': 1.0, 'exposure_ms': 10.0, 'layer': 'BF'},
        'camera',
    ),
    # The LED-tier sibling of the camera cluster: an ownership-scoped off
    # that drove the LED board and frame validity inline on the caller's
    # thread. Its one internal caller (lease release) binds the _impl --
    # teardown runs while a protocol fence is up, where the dispatcher
    # rightly refuses external work.
]

FAMILY_IDS = [f'{family}.{member}' for family, member, _, _ in FAMILIES]

# A member whose body is not named after it: the start members share the
# move bodies with the waited members.
_BODY_OF = {'start_move_absolute': '_move_absolute_impl'}


@pytest.fixture
def executors(sim_scope):
    """The scope's own two lanes, which it built and started.

    Real lanes rather than doubles: the refusal branch is driven through
    protocol_start(), the production state transition, so a
    double would only prove the test agrees with itself. sim_scope is a bare
    scope with no session, the shape of a script's ``Lumascope()``.

    The lanes are the scope's, so a test's fence would outlive
    it into sim_scope's teardown, which stops the stream through the camera
    lane; both are lifted again first.
    """
    lanes = {'io': sim_scope.io_lane(), 'camera': sim_scope.camera_lane()}
    yield lanes
    for lane in lanes.values():
        lane.protocol_end()


def _install_probe(scope, family, member):
    """Bind a recorder over the member's ``_impl`` and return the record.

    The record is the list of thread names the body ran on -- empty when the
    dispatcher never reached it.
    """
    sub = getattr(scope, family)
    threads: list[str] = []

    def _probe(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return IMPL_RESULT

    setattr(sub, _BODY_OF.get(member, f'_{member}_impl'), _probe)
    return sub, threads


def test_a_home_runs_on_the_io_lane(sim_scope):
    """The home, which the caller waits on, runs on the lane too."""
    threads: list[str] = []

    def _home():
        threads.append(threading.current_thread().name)

    with patch.object(sim_scope.motion, '_zhome_impl', _home):
        sim_scope.motion.home('Z')

    assert threads == [sim_scope.io_lane().executor_name]


@pytest.mark.parametrize(('family', 'member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS)
def test_a_shut_lane_refuses_at_once(family, member, kwargs, slot):
    # A shut lane has no worker to drain its queue. Work it accepted would
    # sit there while the caller waited out the member's whole timeout
    # (5 s for an LED or camera write, up to 120 s for a move) and then got
    # a TimeoutError that names nothing; it is refused at submit instead.
    # A scope of its own, not sim_scope: the lane stays shut, and
    # sim_scope's teardown stops the stream through it.
    scope = homed_sim_scope()
    lane = scope.io_lane() if slot == 'io' else scope.camera_lane()
    lane.shutdown()
    sub, threads = _install_probe(scope, family, member)

    started = time.monotonic()
    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        getattr(sub, member)(**kwargs)

    assert time.monotonic() - started < 1.0
    assert excinfo.value.reason == 'scope_disconnected'
    assert excinfo.value.member == member
    assert threads == []


@pytest.mark.parametrize(('family', 'member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS)
def test_protocol_fenced_executor_refuses(sim_scope, executors, family, member, kwargs, slot):
    # A lane a run has fenced refuses before the body runs; a dispatcher
    # that knew only the shut lane would let this through to a silent drop.
    executors[slot].protocol_start()
    sub, threads = _install_probe(sim_scope, family, member)

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        getattr(sub, member)(**kwargs)

    assert excinfo.value.reason == 'exclusive_activity_running'
    assert excinfo.value.member == member
    assert threads == [], f'{family}.{member} ran its body against a protocol-fenced executor'


@pytest.mark.parametrize(('family', 'member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS)
def test_live_executor_submits_and_blocks(sim_scope, executors, family, member, kwargs, slot):
    # sim_scope has no session: the shape of a bare Lumascope() in a script,
    # which used to run every command on the calling thread, unserialized
    # and out of reach of a run holding the scope. Its own lane runs it now.
    sub, threads = _install_probe(sim_scope, family, member)
    worker = executors[slot].executor_name
    caller = threading.current_thread().name

    result = getattr(sub, member)(**kwargs)

    # Returning before the worker has run would leave threads empty here,
    # which is what separates submit-and-block from fire-and-forget.
    assert threads == [worker], (
        f'{family}.{member} must run its body on {worker} and block until it '
        f'has; body ran on {threads}'
    )
    assert caller not in threads
    assert result is IMPL_RESULT


def test_capture_wait_scales_with_the_declared_work(sim_scope, executors):
    """The capture dispatcher's executor wait is a liveness bound, so it
    must sit ABOVE the work the caller declared: base + the content-gate
    retry budget + the summed-frame time + the settle work already
    pending at submit. A flat bound times out a healthy long capture
    (large sum_count at long exposure -- luminescence) while the worker
    is still legitimately grinding, which is a wedged-worker verdict on a
    working capture. Each frame is costed at no less than the camera's
    conservative frame-period floor -- frames cannot arrive faster than
    readout -- and the wait must dominate the body's own
    drain-and-recheck deadline, or a deep legitimate drain surfaces as an
    executor TimeoutError instead of the body's loud None."""
    imaging = sim_scope.imaging
    recorded = {}

    class _RecordingFuture:
        def result(self, timeout=None):
            recorded['timeout'] = timeout
            return None

    with patch.object(executors['camera'], 'put', return_value=_RecordingFuture()):
        imaging.capture_and_wait(timeout_s=5.0, sum_count=10, sum_delay_s=0.2)

    frame_cost = max(
        imaging.exposure_ms_cached / 1000.0,
        imaging._CAPTURE_DEADLINE_MIN_FRAME_PERIOD_S,
    )
    expected = (
        imaging._CAPTURE_WAIT_TIMEOUT_S
        + 5.0
        + 10 * (frame_cost + 0.2)
        + imaging.frame_validity.frames_until_valid()
        * frame_cost
        * imaging._CAPTURE_DEADLINE_MARGIN
    )
    assert recorded['timeout'] == pytest.approx(expected), (
        f'the executor wait must be base + content budget + summed-frame '
        f'time + pending settle work; got {recorded["timeout"]}, expected {expected}'
    )


def test_no_public_imaging_member_writes_the_camera_inline():
    """Shape tripwire for the whole member class, not a spelling grep.

    A public ImagingAPI method that reaches ``_camera_write`` or frame
    invalidation in its own body executes camera-state writes on the
    caller's thread -- unserialized against the camera lane and
    invisible to the protocol fence (an inline body never meets the
    executor's refusal). The dispatch shape puts every such write in a
    ``_impl`` behind ``_dispatch_camera``, so this walks the class and
    fails on any public def whose body touches the write/invalidate
    seams directly. A literal grep for the invalidation spellings
    missed a member of this class once (a public method reaching the
    seams through ``_impl`` calls it hosted inline); the AST walk is
    over the public def's OWN body, so a public that merely dispatches
    stays clean and a future sibling cannot hide.
    """
    import ast

    import tests.ast_seams as ast_seams

    tree = ast_seams.parse_module('modules/lumascope_api/imaging.py')
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == 'ImagingAPI'
    )
    WRITE_SEAMS = {'_camera_write', 'invalidate', 'force_invalidate'}
    offenders = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith('_'):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee = sub.func
                name = (
                    callee.attr
                    if isinstance(callee, ast.Attribute)
                    else callee.id
                    if isinstance(callee, ast.Name)
                    else None
                )
                if name in WRITE_SEAMS:
                    offenders.append(f'{node.name}:{sub.lineno}')
    assert not offenders, (
        'public ImagingAPI members must dispatch camera-state writes, never '
        f'execute them inline in their own body; inline writers found: {offenders}'
    )


def test_no_dispatcher_has_a_lane_less_branch():
    """Every scope builds its lanes, so a branch for a missing one is dead --
    and a dead branch that runs the body on the calling thread is the
    unserialized, unfenceable path this contract exists to close. AST walk,
    so a respelling of the test cannot hide one."""
    import ast

    import tests.ast_seams as ast_seams

    lane_names = {'ex', 'executor', '_io_executor', '_camera_executor'}
    offenders = []
    for path in (
        'modules/lumascope_api/illumination.py',
        'modules/lumascope_api/imaging.py',
        'modules/lumascope_api/motion.py',
        'modules/lumascope_api/_lumascope.py',
    ):
        for node in ast.walk(ast_seams.parse_module(path)):
            if not isinstance(node, ast.Compare):
                continue
            sides = [node.left, *node.comparators]
            if not any(isinstance(x, ast.Constant) and x.value is None for x in sides):
                continue
            for side in sides:
                name = side.attr if isinstance(side, ast.Attribute) else getattr(side, 'id', None)
                if name in lane_names:
                    offenders.append(f'{path}:{node.lineno}')
    assert not offenders, f'a dispatcher tests for a missing lane: {offenders}'
