# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Capability-probe reality guard: a name probed on the scope must exist.

`hasattr(mock, 'anything')` is True, and a hand-rolled fake is written to
make the code under test proceed. Together they let a production branch
that can NEVER run stay green forever -- the suite proves the fake world.
Two such families were confirmed by the 2026-08 census:

    modules/video_capture.py    hasattr(self._scope, 'led_on_fast')
    modules/metrics_logger.py   getattr(self._scope, 'camera', None)

Neither name exists on `Lumascope`. `led_on_fast` lives on the
`illumination` sub-API (the guard probes the parent and then calls
through the child, so it is False in production and the else-branch is
what runs); `camera` was never assigned at all, which leaves the frame-
flow stall watchdog permanently disarmed. Fixing only those four sites
would leave the ENABLER, so this guard targets the class.

Oracle: `dir()` of a CONSTRUCTED `Lumascope(simulate=True)`, checked
RECEIVER-SPECIFICALLY.

Both halves of that sentence are load-bearing and each killed an earlier
design:

- CONSTRUCTED, not the class. The six sub-APIs, the driver slots and
  `metrics_logger` are all assigned in `__init__`, so a class-level
  oracle sees none of them -- it would miss Family 2 and false-positive
  on every legitimate instance-attribute probe.
- RECEIVER-SPECIFIC, not the union of scope + sub-APIs. A union oracle
  accepts `led_on_fast` because `IlluminationAPI` has it, so it would
  have PASSED Family 1 -- the exact bug this guard exists to catch.

Coverage, stated honestly (Rule 39 -- this guard is not a probe-site
audit). Measured at introduction: 256 literal-name probe sites across
`modules/` + `ui/` (89 `hasattr`, 167 three-arg `getattr`). Of those, 25
have the scope object as receiver -- the subset this oracle can
adjudicate -- and 13 probe a DRIVER through the scope
(`scope._camera_driver`, `self._led_driver`). The rest probe widgets,
tasks and duck-typed objects.

The driver sites are excluded on purpose, not by oversight. There is
exactly one `Lumascope` class, so a name it lacks is unambiguously dead;
a driver is legitimately polymorphic -- `timestamp_tick_frequency_hz`,
`cam_image_handler`, `model_name` and `_device_serial` are Pylon/IDS
specific, and `hasattr(driver, 'disconnect')` guards the Null boards --
so "absent from the simulated driver" does NOT mean "dead", and a
sim-instance oracle would fire falsely on real capability probes.
Guarding that surface needs a union-over-concrete-drivers oracle; those
13 sites are homed to the deferred suite-quality batch alongside the
camera-driver parity extension (PylonCamera/IDSCamera are never
compared today). `test_probe_scan_is_not_vacuous` below keeps this
guard from silently shrinking to zero coverage.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from tests.ast_seams import iter_package_modules

# Packages that hold callers of the scope API. `drivers/` is excluded on
# purpose: it sits BELOW the API and never holds a scope reference.
SCANNED_PACKAGES = ('modules', 'ui')

# Receiver expressions denoting the Lumascope object. Matched on the
# FINAL attribute so dotted chains are covered without enumerating them
# (`self._scope`, `ctx.scope`, `_app_ctx.ctx.scope`), while near-misses
# like `self.scope_display` are not.
_SCOPE_TAIL_NAMES = frozenset({'scope', '_scope'})

# Names allowed to be missing, keyed by NAME so the entry survives the
# line shifts of any merge. Each entry names the owner that retires it --
# an allowlist without an expiry is a permanent exemption.
_ALLOWED_MISSING = {
    'camera': (
        'Family 2. R1 decision D1=C (Eric, 2026-08-07): the frame-flow '
        'stall detector stays DISARMED and R3 owns arming it with a '
        'correct reconnect lifecycle plus an ImagingAPI-owned fps truth '
        'source. Arming it today would also activate a dormant Rule 1 '
        'read of a UI widget private attribute. Expires with R3.'
    ),
}

# Floor for the not-vacuous self-check. Set from the measured count at
# introduction, well below it, so ordinary code churn does not trip it
# but a scanner that has stopped matching anything does.
_MIN_SCOPE_PROBE_SITES = 12


@dataclass(frozen=True)
class ProbeSite:
    """One `hasattr`/3-arg-`getattr` call with a literal attribute name."""

    rel_path: str
    lineno: int
    func: str
    receiver: str
    name: str

    def __str__(self) -> str:
        return f'{self.rel_path}:{self.lineno}: {self.func}({self.receiver}, {self.name!r})'


def _receiver_is_scope(receiver: str) -> bool:
    """True when the receiver expression denotes the Lumascope object."""
    return receiver.split('.')[-1] in _SCOPE_TAIL_NAMES


def _probe_name(node: ast.Call) -> str | None:
    """Return the literal attribute name probed, or None.

    Recognizes `hasattr(x, 'n')` and the 3-arg `getattr(x, 'n', default)`
    only. Two-arg `getattr` raises on a missing name, so it is an access,
    not a probe. A computed name is not something a static scan can
    resolve; those are counted, never asserted on.
    """
    if not isinstance(node.func, ast.Name):
        return None
    arity = {'hasattr': 2, 'getattr': 3}.get(node.func.id)
    if arity is None or len(node.args) != arity:
        return None
    name_arg = node.args[1]
    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        return name_arg.value
    return None


def iter_probe_sites(packages=SCANNED_PACKAGES):
    """Yield every literal-name capability probe under `packages`."""
    for rel_path, tree in iter_package_modules(packages):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _probe_name(node)
            if name is None:
                continue
            yield ProbeSite(
                rel_path=rel_path,
                lineno=node.lineno,
                func=node.func.id,
                receiver=ast.unparse(node.args[0]),
                name=name,
            )


def scope_probe_sites():
    """The subset this guard's oracle can adjudicate."""
    return [site for site in iter_probe_sites() if _receiver_is_scope(site.receiver)]


@pytest.fixture(scope='module')
def scope_surface():
    """Names present on a fully constructed Lumascope.

    Constructed with production defaults -- `register_metrics` and
    `register_atexit` left ON -- because production code probes the
    fully-wired object, and an oracle built from a cheaper construction
    would report attributes missing that production callers really do
    see.
    """
    from modules.lumascope_api import Lumascope

    scope = Lumascope(simulate=True)
    try:
        yield frozenset(dir(scope))
    finally:
        scope.disconnect()


def test_scope_capability_probes_name_real_attributes(scope_surface):
    """Every name probed on the scope exists on the scope itself.

    A failure here means one of two things, and both are real defects:
    the probed branch is dead code (the name never existed), or the name
    moved to a sub-API and the probe was not updated -- in which case the
    guard is False forever and the fallback branch is the only one that
    ever runs.
    """
    violations = [
        site
        for site in scope_probe_sites()
        if site.name not in scope_surface and site.name not in _ALLOWED_MISSING
    ]
    assert not violations, (
        'Capability probe(s) name an attribute that Lumascope does not have, '
        'so the guarded branch can never run:\n  '
        + '\n  '.join(str(site) for site in violations)
        + '\n\nFix the probe (usually: probe the sub-API that owns the name, '
        'or delete the dead branch). Only add to _ALLOWED_MISSING with an '
        'owner and an expiry.'
    )


def test_oracle_catches_the_known_wrong_world_families(scope_surface):
    """The oracle bites: it rejects every name the census confirmed dead.

    This is the guard's own fail-before proof, kept permanently rather
    than run once. With the allowlist ignored, the confirmed site must be
    reported -- the `camera` probe in metrics_logger. The `led_*_fast`
    probes that used to sit alongside it lived in video_capture, which no
    longer exists; their expectations retired with the file.

    The second half is the part that matters most. `led_on_fast` DOES
    exist on the `illumination` sub-API, so an oracle built from the
    union of the scope and its sub-APIs would accept the probe and this
    whole guard would be decorative. Asserting the union would have been
    fooled locks the receiver-specific design in place: if someone
    "simplifies" the oracle to a union, this test fails.
    """
    from modules.lumascope_api import Lumascope

    caught = {
        (site.rel_path, site.name) for site in scope_probe_sites() if site.name not in scope_surface
    }
    expected = {
        ('modules/metrics_logger.py', 'camera'),
    }
    missed = expected - caught
    assert not missed, (
        f'The oracle no longer rejects known-dead probes {sorted(missed)}. '
        f'Either the probe was fixed (delete the expectation AND its '
        f'_ALLOWED_MISSING entry) or the oracle stopped biting.'
    )

    scope = Lumascope(simulate=True)
    try:
        union = set(dir(scope))
        for sub_api in ('illumination', 'imaging', 'motion', 'diagnostics', 'io'):
            union |= set(dir(getattr(scope, sub_api)))
    finally:
        scope.disconnect()
    assert 'led_on_fast' in union, (
        'Premise check: led_on_fast is supposed to exist on a sub-API. If it '
        'no longer does, the union-oracle hazard is gone and this assertion '
        'should be retired.'
    )


def test_probe_scan_is_not_vacuous():
    """The scan still finds scope probes at all.

    Without this, renaming `_scope` or restructuring the packages would
    silently reduce the guard above to an assertion over an empty list --
    a green test proving nothing. Failing loudly here is the point.
    """
    found = scope_probe_sites()
    assert len(found) >= _MIN_SCOPE_PROBE_SITES, (
        f'Found only {len(found)} scope-receiver probe sites, expected at '
        f'least {_MIN_SCOPE_PROBE_SITES}. The scan is no longer matching '
        f'production code -- check _SCOPE_TAIL_NAMES and SCANNED_PACKAGES '
        f'against how the scope reference is spelled now.'
    )


def test_allowlist_entries_are_still_needed(scope_surface):
    """Retire an allowlist entry the moment its name becomes real.

    Each entry exempts a KNOWN wrong-world site. When its owner lands the
    fix, the name starts existing (or the probe goes away) and the entry
    becomes a lie that would mask the next regression of the same shape.
    """
    now_real = sorted(name for name in _ALLOWED_MISSING if name in scope_surface)
    assert not now_real, (
        f'These _ALLOWED_MISSING entries now exist on Lumascope and must be '
        f'deleted from the allowlist: {now_real}'
    )

    probed = {site.name for site in scope_probe_sites()}
    unused = sorted(name for name in _ALLOWED_MISSING if name not in probed)
    assert not unused, (
        f'These _ALLOWED_MISSING entries no longer match any probe site and '
        f'must be deleted: {unused}'
    )
