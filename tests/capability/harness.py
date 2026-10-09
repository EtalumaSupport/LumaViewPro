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

THE TARGET. A probe asks its question of the simulator unless it is run with
`--hardware` on its command line, and then of the scope connected to this
machine, through `hardware_session()`. That is the only door to hardware. A
simulated constructor refuses under `--hardware`, so a probe with no hardware
path can never print a simulator's verdict for a hardware question. The suite
never passes the flag.
"""

import contextlib
import json
import os
import pathlib
import sys
import tempfile
import time
import traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Probes resolve their relative paths (data/, output folders) against the repo root.
os.chdir(REPO)

# Each probe subprocess is handed its own scratch root, so concurrent probes
# never share a live_folder or data/current.json.
SCRATCH = pathlib.Path(
    os.environ.get('LVP_CAPABILITY_SCRATCH') or tempfile.mkdtemp(prefix='lvp_capability_')
)
SCRATCH.mkdir(parents=True, exist_ok=True)

TILING_JSON = REPO / 'data' / 'tiling.json'

HARDWARE = '--hardware' in sys.argv[1:]
PROBE = pathlib.Path(sys.argv[0]).stem

# (name, passed, kind) where kind is 'check' or 'void'.
RESULTS: list[tuple[str, bool, str]] = []

# What a probe measured, by name, for the reader to judge (`figure`).
FIGURES: dict[str, object] = {}

# The connected scope's identity, set by `hardware_session()` at bring-up.
IDENTITY: dict[str, object] = {}


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


def figure(name, value):
    """A measurement the probe reports without judging it."""
    FIGURES[name] = value
    print(f'FIGURE {name} = {value}', flush=True)
    return value


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
    failed = bool(failed_checks or filled_voids or bad)
    print(f'TARGET: {"hardware" if HARDWARE else "simulator"}', flush=True)
    if HARDWARE:
        _write_verdict(failed)
    return 1 if failed else 0


def _write_verdict(failed):
    """Leave the hardware run's verdict where the reader can find it.

    Stamped with the build and the scope it ran on, so the file answers on
    its own which code asked which instrument.
    """
    import lvp_logger

    verdict = {
        'probe': PROBE,
        'target': 'hardware',
        'passed': not failed,
        'lvp_commit': lvp_logger.git_revision(),
        'scope': IDENTITY,
        'checks': [{'name': n, 'passed': p, 'kind': k} for n, p, k in RESULTS],
        'figures': FIGURES,
        'log_folder': lvp_logger.log_dir,
    }
    path = SCRATCH / f'{PROBE}_verdict.json'
    path.write_text(json.dumps(verdict, indent=2, default=str))
    print(f'VERDICT FILE: {path}', flush=True)
    print(f'LOG FOLDER: {lvp_logger.log_dir}', flush=True)


def _refuse_hardware():
    """A simulated session asked for under `--hardware` is refused, by name."""
    if HARDWARE:
        print(
            f'REFUSED: {PROBE} has no hardware target -- it builds a simulated session, '
            'so under --hardware its verdict would describe the simulator.',
            flush=True,
        )
        sys.exit(2)


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
    from modules.scope_session import ScopeSession

    from tests.settings_fixtures import complete_settings

    _refuse_hardware()
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
        # A raise is a failed check, not only a crash: the traceback, then the
        # verdict it leaves.
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

    _refuse_hardware()
    overrides = {'live_folder': str(live_folder), 'stage_offset': {'x': 0.0, 'y': 0.0}}
    for layer in common_utils.get_layers():
        overrides[layer] = {
            # 'image' / 'none' is the stored vocabulary; a bool reads as 'off'.
            'acquire': 'image' if layer in acquiring else None,
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
        session.shutdown()


@contextlib.contextmanager
def hardware_session():
    """The scope connected to this machine, brought up the way any script does.

    Only under `--hardware`. The settings are the installation's own
    `current.json`, read from the root the app reads: a root without one is
    refused, because the shipped template would configure a scope that is not
    the one on the desk. Bring-up is the one host bring-up,
    `start_application_session`, which returns once the stage has homed; a
    scope with no motor board comes straight up. An objective nobody has
    confirmed stops the probe -- it never tells the scope what is in the
    light path. Nothing here saves settings, and the file is checked
    unchanged at the end.
    """
    if not HARDWARE:
        print(f'REFUSED: hardware_session() is the --hardware door; {PROBE} was run without it.')
        sys.exit(2)

    from modules.path_utils import get_source_root
    from modules.scope_session import ScopeSession

    root = get_source_root()
    current = root / 'data' / 'current.json'
    if not current.is_file():
        print(
            f'REFUSED: no {current} -- the session would come up on the shipped template, '
            'not this installation. Start LumaViewPro once on this machine first.',
            flush=True,
        )
        sys.exit(2)
    stamp = current.stat().st_mtime_ns
    print(f'SETTINGS: {current}', flush=True)
    print(
        'BRING-UP: homes the stage and turns the turret to slot 1; a layer with autofocus on '
        'moves Z during a run. Close LumaViewPro first: it holds the camera.',
        flush=True,
    )

    session = ScopeSession.create(ScopeSession.load_user_settings(str(root)), source_path=str(root))
    try:
        session.start_application_session()
        scope = session.scope
        IDENTITY.update(
            model=scope.layer_identity.model,
            motor=scope.diagnostics.get_motor_info(),
            camera={
                'model': scope.capabilities.camera_model,
                'serial': scope.capabilities.camera_serial_number,
                'tick_hz': scope.capabilities.camera_timestamp_tick_hz,
            },
        )
        print(f'SCOPE: {IDENTITY}', flush=True)
        question = session.objective_question()
        if question is not None:
            print(
                f'STOPPED: the objective is not confirmed ({question}). '
                'Confirm it in LumaViewPro, close it, and run the probe again.',
                flush=True,
            )
            sys.exit(2)
        runner = session.create_protocol_runner()
        yield session, runner
    finally:
        with contextlib.suppress(BaseException):
            session.shutdown()
        check("the installation's current.json is unchanged", current.stat().st_mtime_ns == stamp)
