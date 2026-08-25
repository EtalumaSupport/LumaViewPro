# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scope doubles that cannot answer for attributes the real scope lacks.

A bare `MagicMock()` scope says yes to everything: `hasattr` is True for
every name, every attribute access invents a child mock, and every call
returns one. A test written against it passes whether or not the code
under test asks the real `Lumascope` for something it has. Two families
of production-dead code stayed green in this suite for exactly that
reason (see `tests/test_capability_probe_reality.py`).

`spec_scope()` builds its double by autospec'ing a CONSTRUCTED
`Lumascope(simulate=True)` INSTANCE. Accessing a name the real scope
does not have raises `AttributeError` instead of inventing a child, and
calling a method with the wrong signature raises `TypeError`.

Autospec the INSTANCE, never the class. All six sub-APIs
(`illumination`, `imaging`, `motion`, `diagnostics`, `io`,
`runtime_state`), the driver slots and `metrics_logger` are assigned in
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

from unittest.mock import create_autospec


def build_real_sim_scope():
    """A constructed `Lumascope(simulate=True)`, for use as the spec.

    Production defaults are kept (`register_metrics` / `register_atexit`
    left on) so the spec covers everything a production caller can
    reach, including `metrics_logger`.

    The caller owns disconnecting it. `spec_scope()` does that for you.
    """
    from modules.lumascope_api import Lumascope

    return Lumascope(simulate=True)


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

    The caller owns disconnecting it.
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
        if not scope.motion._home_impl():
            raise AssertionError('the simulated scope failed to home')
    finally:
        driver.set_timing_mode(prior_timing)
    return scope


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
        scope.illumination.led_on(channel=0, mA=100)   # ok
        scope.led_on_fast(channel=0, mA=100)           # AttributeError
    """
    real = build_real_sim_scope()
    try:
        double = create_autospec(real, instance=True, spec_set=True)
    finally:
        # The spec is captured by create_autospec; the live object has
        # served its purpose and must not keep simulated drivers open.
        real.disconnect()

    for name, value in attrs.items():
        # spec_set makes this raise AttributeError for a name the real
        # scope lacks -- the point of the fixture, so it is not caught.
        setattr(double, name, value)
    return double
