# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The dispatch contract for the public hardware members.

Each public hardware member is a DISPATCHER over a private ``_impl`` that
holds the body. Every internal caller and both public tiers bind ``_impl``
directly, so the dispatcher is reached only by external callers -- an SDK
script, a REST handler, a MATLAB client -- who are never on an executor
worker and never on the protocol or autofocus thread.

Three branches, and this file pins one test per branch:

  * no executor registered -> run ``_impl`` on the calling thread. A bare
    ``Lumascope()`` in a script or an example has no executors and must
    still drive hardware.
  * the executor will not accept work -> raise HardwareCommandRefusedError.
  * otherwise -> submit to the executor and block for the result.

The middle branch asks WHETHER the executor accepts work, never WHY it
might not. A run disables the camera executor while io and file are fenced
by protocol_start(); ``put()`` returns None for both states and the caller
cannot tell them apart. A branch keyed on "is a protocol running" lets a
camera write submit, receive None, and silently reach no hardware -- so
both states are driven here and both must refuse identically.

Parametrized across all three families because they do not share a dispatch
shape: illumination submits through ``_submit_io``, imaging and motion
construct their IOTask directly, callback plumbing differs, and
``wait_until_complete`` exists only on motion. A contract pinned on one
family would not catch the other two diverging from it.

The probe replaces ``_impl`` with a recorder rather than asserting hardware
side effects: what is under test is WHERE and WHETHER the body runs, not
what the body does. The recorder's captured thread name is the evidence --
the calling thread for a direct call, the executor's worker for a submit.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.sequential_io_executor import SequentialIOExecutor

# Sentinel returned by the probe. The dispatcher must hand it back on both
# non-refusing branches: a direct call returns what _impl returned, and a
# submit-and-block returns what the worker produced. A dispatcher that
# submits without waiting returns None here and fails.
IMPL_RESULT = object()

# (family, member, async member, kwargs, which executor carries it)
#
# The async member's name does not always track the base name
# (move_absolute dispatches through move_absolute_async), so it is
# carried explicitly rather than derived by suffix. The camera family
# carries None: its fire-and-forget tier was measured consumerless and
# deleted rather than collapsed, so the camera capability ends with the
# dispatcher as its only public form and has no async member to pin.
FAMILIES = [
    ('illumination', 'led_on', 'led_on_async', {'channel': 0, 'mA': 10.0}, 'io'),
    ('imaging', 'set_gain_db', None, {'gain_db': 1.0}, 'camera'),
    (
        'motion',
        'move_absolute',
        'move_absolute_async',
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
        None,
        {
            'state': True,
            'settings': {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0},
        },
        'camera',
    ),
    ('imaging', 'set_auto_exposure_time', None, {'state': True}, 'camera'),
    ('imaging', 'set_frame_size', None, {'w': 640, 'h': 480}, 'camera'),
    ('imaging', 'set_binning_size', None, {'size': 1}, 'camera'),
    ('imaging', 'set_pixel_format', None, {'pixel_format': 'Mono8'}, 'camera'),
    ('imaging', 'set_conversion_gain_mode', None, {'mode': 'High'}, 'camera'),
    ('imaging', 'set_line_noise_reduction', None, {'enabled': True}, 'camera'),
    (
        'imaging',
        'update_auto_gain_target_brightness',
        None,
        {'target_brightness': 0.5},
        'camera',
    ),
    (
        'imaging',
        'auto_gain_once',
        None,
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
        None,
        {'gain_db': 1.0, 'exposure_ms': 10.0},
        'camera',
    ),
    # The LED-tier sibling of the camera cluster: an ownership-scoped off
    # that drove the LED board and frame validity inline on the caller's
    # thread. Its one internal caller (lease release) binds the _impl --
    # teardown runs while a protocol fence is up, where the dispatcher
    # rightly refuses external work.
    ('illumination', 'leds_off_owned', None, {'owner': 'testowner'}, 'io'),
]

FAMILY_IDS = [f'{family}.{member}' for family, member, _, _, _ in FAMILIES]

ASYNC_FAMILIES = [f for f in FAMILIES if f[2] is not None]
ASYNC_FAMILY_IDS = [f'{family}.{member}' for family, member, _, _, _ in ASYNC_FAMILIES]


@pytest.fixture
def executors(sim_scope):
    """Real executors, started and registered on the scope.

    Real SequentialIOExecutors rather than doubles: the refusal branches are
    driven through disable() and protocol_start(), the production state
    transitions, so a double would only prove the test agrees with itself.
    """
    io = SequentialIOExecutor(name='TEST_IO')
    camera = SequentialIOExecutor(name='TEST_CAMERA')
    file_io = SequentialIOExecutor(name='TEST_FILE')
    for ex in (io, camera, file_io):
        ex.start()
    sim_scope.register_executors(camera_executor=camera, io_executor=io, file_io_executor=file_io)
    yield {'io': io, 'camera': camera, 'file_io': file_io}
    for ex in (io, camera, file_io):
        ex.shutdown()


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

    setattr(sub, f'_{member}_impl', _probe)
    return sub, threads


@pytest.mark.parametrize(
    ('family', 'member', 'async_member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS
)
def test_absent_executor_runs_impl_on_the_calling_thread(
    sim_scope, family, member, async_member, kwargs, slot
):
    # sim_scope registers no executors, which is the shape of a bare
    # Lumascope() in a script. This branch PRESERVES what the base member
    # does today -- it never touches an executor -- through the dispatcher
    # that now fronts it. What changes is the absorbed blocking tier, which
    # raises RuntimeError here today rather than driving the hardware.
    assert sim_scope._io_executor is None
    assert sim_scope._camera_executor is None

    sub, threads = _install_probe(sim_scope, family, member)
    caller = threading.current_thread().name

    result = getattr(sub, member)(**kwargs)

    assert threads == [caller], (
        f'{family}.{member} with no executor must run its body on the calling '
        f'thread, not raise and not defer; body ran on {threads}'
    )
    assert result is IMPL_RESULT


@pytest.mark.parametrize(
    ('family', 'member', 'async_member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS
)
def test_disabled_executor_refuses(
    sim_scope, executors, family, member, async_member, kwargs, slot
):
    executors[slot].disable()
    sub, threads = _install_probe(sim_scope, family, member)

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        getattr(sub, member)(**kwargs)

    assert excinfo.value.reason == 'exclusive_activity_running'
    assert excinfo.value.member == member
    assert threads == [], (
        f'{family}.{member} ran its body against a disabled executor; the '
        f'refusal must precede the work'
    )


@pytest.mark.parametrize(
    ('family', 'member', 'async_member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS
)
def test_protocol_fenced_executor_refuses(
    sim_scope, executors, family, member, async_member, kwargs, slot
):
    # The other half of the middle branch. A fenced executor is NOT a
    # disabled one -- a run fences io and file while disabling camera -- and
    # a dispatcher that only knows about disable() lets this one through to
    # a silent drop.
    executors[slot].protocol_start()
    sub, threads = _install_probe(sim_scope, family, member)

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        getattr(sub, member)(**kwargs)

    assert excinfo.value.reason == 'exclusive_activity_running'
    assert excinfo.value.member == member
    assert threads == [], f'{family}.{member} ran its body against a protocol-fenced executor'


@pytest.mark.parametrize(
    ('family', 'member', 'async_member', 'kwargs', 'slot'), FAMILIES, ids=FAMILY_IDS
)
def test_live_executor_submits_and_blocks(
    sim_scope, executors, family, member, async_member, kwargs, slot
):
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


@pytest.mark.parametrize(
    ('family', 'member', 'async_member', 'kwargs', 'slot'),
    ASYNC_FAMILIES,
    ids=ASYNC_FAMILY_IDS,
)
def test_async_warns_when_the_executor_drops_it(
    sim_scope, executors, family, member, async_member, kwargs, slot
):
    # The async tiers stay fire-and-forget -- they must NOT raise, or every
    # UI callsite would need a handler for a state it cannot prevent. But a
    # dropped submit has to leave a trace: today put() returns None and the
    # submit helper reports success anyway, so the command vanishes with no
    # record that it was ever issued.
    executors[slot].disable()
    sub, _threads = _install_probe(sim_scope, family, member)
    module = type(sub).__module__

    with patch(f'{module}.logger') as mock_logger:
        getattr(sub, async_member)(**kwargs)

    warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)
    assert async_member in warned, (
        f'{family}.{async_member} was dropped by the executor without a '
        f'warning naming it; warnings seen: {warned!r}'
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
