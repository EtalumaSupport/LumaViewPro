# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The one harness for the capability probes.

A probe answers a single question: can a SCRIPT do what the GUI does, going
through `ScopeSession` / `modules.lumascope_api` only? The probe's outcome is
the classification -- every verdict here was produced by a run, never a read.

Two verbs, and the second is why these probes can live in the repo at all:

  `check(name, cond)`  the capability works; it must PASS.
  `void(name, cond, reason)`  a known hole; it is EXPECTED TO FAIL.

A void that starts passing is progress, and it fails the run demanding its pin
be lowered -- the same shape `tests/guards/test_architecture_fixes.py` uses for
the reach proxies. Without the second verb a probe that correctly documents a
hole can only ever be red, which is why the census that produced these probes
had to leave them outside the suite.

Probes run as SUBPROCESSES (see `test_capability_probes.py`). That is not a
style choice: `tests/conftest.py` installs Kivy mocks at import time, and the
load-bearing assertion here is that Kivy is ABSENT from `sys.modules`.
"""

import contextlib
import os
import pathlib
import sys
import tempfile
import traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# The session resolves its shipped data files against the CWD (source_path='.').
os.chdir(REPO)

# Each probe subprocess is handed its own scratch root, so concurrent probes
# never share a live_folder or data/current.json.
SCRATCH = pathlib.Path(
    os.environ.get('LVP_CAPABILITY_SCRATCH') or tempfile.mkdtemp(prefix='lvp_capability_')
)
SCRATCH.mkdir(parents=True, exist_ok=True)

TILING_JSON = REPO / 'data' / 'tiling.json'

# (name, passed, kind) where kind is 'check' or 'void'.
RESULTS: list[tuple[str, bool, str]] = []


def check(name, cond, detail=''):
    """A capability that must work. Recorded PASS/FAIL."""
    RESULTS.append((name, bool(cond), 'check'))
    print(('PASS ' if cond else 'FAIL ') + name + (f'   [{detail}]' if detail else ''), flush=True)
    return bool(cond)


def void(name, cond, reason):
    """A known hole: a script cannot do this, and the probe proves it.

    `cond` is the same condition `check` would take -- TRUE means the
    capability works. A void is satisfied when it is FALSE.
    """
    RESULTS.append((name, bool(cond), 'void'))
    state = 'VOID FILLED' if cond else 'VOID'
    print(f'{state} {name}   [{reason}]', flush=True)
    return bool(cond)


# `ok` is the settings slice's spelling of `check`; one implementation, two
# names, so the rescued probes keep their call sites (Rule 35).
ok = check


def banner(text):
    print('\n' + '=' * 8 + ' ' + text + ' ' + '=' * 8, flush=True)


def imported_ui_modules():
    """Kivy / `ui.*` modules that reached `sys.modules`. Empty is the contract."""
    return [
        m
        for m in sys.modules
        if m == 'kivy' or m.startswith('kivy.') or m == 'ui' or m.startswith('ui.')
    ]


def assert_no_ui():
    """The contract a capability probe exists to hold: it drove modules/ only.

    Only one of the five slice harnesses this replaced had this; the other
    four relied on nobody writing the import. `report()` checks it for every
    probe now, and this stays for the probes that state it inline.
    """
    bad = imported_ui_modules()
    print('UI/KIVY MODULES IMPORTED:', bad if bad else 'NONE', flush=True)
    return not bad


def report():
    """Print the verdict and return the process exit code.

    Green means every `check` passed AND every `void` still fails. A filled
    void is a FAILURE here on purpose: it is the ratchet demanding its pin be
    lowered in the same commit that filled it.
    """
    checks = [(n, p) for n, p, k in RESULTS if k == 'check']
    voids = [(n, p) for n, p, k in RESULTS if k == 'void']
    failed_checks = [n for n, p in checks if not p]
    filled_voids = [n for n, p in voids if p]

    bad = imported_ui_modules()
    print(f'\nUI/KIVY MODULES IMPORTED: {bad if bad else "NONE"}', flush=True)
    print(
        f'=== {len(checks) - len(failed_checks)}/{len(checks)} checks passed; '
        f'{len(voids) - len(filled_voids)}/{len(voids)} voids still open ===',
        flush=True,
    )
    for name in failed_checks:
        print(f'FAILED CHECK: {name}', flush=True)
    for name in filled_voids:
        print(
            f'VOID FILLED: {name} -- the capability now works. Lower the pin in this commit.',
            flush=True,
        )
    if bad:
        print(f'UI LEAK: a probe imported {bad}; a capability probe drives modules/ only.')
    return 1 if (failed_checks or filled_voids or bad) else 0


def live_dir(name):
    """A fresh live_folder for one probe."""
    d = SCRATCH / name
    d.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=f'{name}_', dir=str(d)))


# The post-processing slice's spelling of the same thing.
def probe_dir(name):
    return live_dir(name)


def make_session(name='probe', *, home=False, **overrides):
    """A headless simulated session, built the way an L2 caller builds one.

    `home` runs `ScopeSession.start_application_session()`, the one bring-up
    seam, and waits for the queued home to drain. Without it every protocol
    move is refused with `AxisStateUnknownError` and the run dies at
    `consecutive_scan_failures`, so a protocol probe MUST ask for it.
    """
    import time

    from modules.scope_session import ScopeSession

    from tests.settings_fixtures import complete_settings

    live = live_dir(name)
    session = ScopeSession.create(
        complete_settings(live_folder=str(live), **overrides), simulate=True
    )
    if home:
        session.start_application_session()
        deadline = time.time() + 60
        while time.time() < deadline and not session.scope.motion.has_homed():
            time.sleep(0.25)
    return session, live


def new_session(**overrides):
    """The session alone, for a probe that does not need its live_folder."""
    session, _live = make_session(**overrides)
    return session


def run(body):
    """Run a probe body against a fresh session and exit with its verdict."""
    session = new_session()
    try:
        body(session)
    except BaseException:
        # lvp_logger installs a sys.excepthook that sends an uncaught traceback
        # to a log file, so a probe that raises would otherwise exit in silence.
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        with contextlib.suppress(BaseException):
            session.shutdown()
    sys.exit(report())


@contextlib.contextmanager
def headless_session(live_folder, acquiring=('BF', 'Blue'), **extra):
    """A homed session with its executors up and a protocol runner attached."""
    import modules.common_utils as common_utils
    from modules.scope_session import ScopeSession

    from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
    from tests.settings_fixtures import complete_settings

    overrides = {'live_folder': str(live_folder), 'stage_offset': {'x': 0.0, 'y': 0.0}}
    for layer in common_utils.get_layers():
        overrides[layer] = {
            # 'image' / 'none' is the stored vocabulary; a bool reads as 'off'.
            'acquire': 'image' if layer in acquiring else 'none',
            'composite_brightness_threshold': 25,
        }
    overrides['turret_objectives'] = dict(TEST_TURRET_OBJECTIVES)
    overrides.update(extra)

    session = ScopeSession.create(complete_settings(**overrides), simulate=True)
    scope = session.scope
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    home_sim_scope(scope)
    runner = session.create_protocol_runner()
    try:
        yield session, runner
    finally:
        runner.shutdown()
        session.shutdown()
