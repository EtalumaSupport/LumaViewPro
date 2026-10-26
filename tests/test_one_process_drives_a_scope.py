# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One process drives an installation's scope, and a process drives one real scope.

``modules.lvp_lock.take_instance_lock`` is a host's take, made before the
GUI or the REST server loads anything that opens the scope: another
process's take is answered None, so ``python -m rest`` (``rest.__main__``)
refuses to start while one is held. Every real scope takes the same lock
where its hardware opens (``take_scope_lock``, from ``_init_minimal``),
however it is built: refused with ``ScopeInUseError`` while another process
holds it or another real scope in this process is open, with nothing
opened; given back at the scope's ``disconnect()``, whatever that raised,
or when its constructor raises. A simulated scope takes none.

Each holder is an installation of its own on a free port; another process
is a subprocess holding the port.
"""

import contextlib
import json
import socket
import subprocess
import sys
import threading

import pytest

import modules.lumascope_api._lumascope as lumascope_module
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.registry import DriverFallback, NoLiveDriverError
from modules.exceptions import InstallationFileError, ScopeDisconnectError, ScopeInUseError
from modules.lumascope_api import Lumascope
from modules.lvp_lock import take_instance_lock
from modules.path_utils import get_source_root
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings, installation_of_its_own

# The server's test takes the installation's own lock, so the file runs on
# one worker (`--dist loadgroup`, in the pytest addopts).
pytestmark = pytest.mark.xdist_group('installation_lock')

# Binds the port it is given and holds it until its stdin closes; says
# 'held', or 'taken' when a process outside the test already holds it.
_HOLD = (
    'import socket, sys\n'
    's = socket.socket()\n'
    'try:\n'
    "    s.bind(('127.0.0.1', int(sys.argv[1])))\n"
    "    print('held', flush=True)\n"
    'except OSError:\n'
    "    print('taken', flush=True)\n"
    'sys.stdin.read()\n'
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _installation_on(port: int):
    root = installation_of_its_own()
    (root / 'data' / 'current.json').write_text(json.dumps({'lvp_lock_port': port}))
    return root


@contextlib.contextmanager
def _held_by_another_process(port: int):
    holder = subprocess.Popen(
        [sys.executable, '-c', _HOLD, str(port)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        said = holder.stdout.readline().decode().strip()
        assert said in ('held', 'taken'), said
        yield said
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)


def _another_process_can_take(port: int) -> bool:
    with _held_by_another_process(port) as said:
        return said == 'held'


@pytest.fixture
def no_hardware(monkeypatch):
    """The registries answer as a machine with no scope does: null boards, no camera."""
    absent = DriverFallback('not_detected', ())
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **kw: (NullLEDBoard(), absent),
    )
    monkeypatch.setattr(
        lumascope_module.motor_registry,
        'create_with_fallback',
        lambda name='auto', **kw: (NullMotionBoard(), absent),
    )

    def no_camera(name='auto', **kw):
        if kw.get('simulate'):
            return real_create(name, **kw)
        raise NoLiveDriverError('camera', absent)

    real_create = lumascope_module.camera_registry.create
    monkeypatch.setattr(lumascope_module.camera_registry, 'create', no_camera)


def _scope_threads() -> set[threading.Thread]:
    """The scope threads running now, as objects: another test's leftover has the same name."""
    return {
        t
        for t in threading.enumerate()
        if t.name in ('IO_WORKER', 'CAMERA_WORKER', 'motion-monitor')
    }


def test_a_host_take_is_answered_none_while_another_process_holds_the_lock():
    port = _free_port()
    root = _installation_on(port)
    with _held_by_another_process(port):
        assert take_instance_lock(root) is None

    taken = take_instance_lock(root)
    assert taken is not None
    taken.close()


def test_every_real_scope_is_refused_while_another_process_holds_the_lock(no_hardware):
    port = _free_port()
    root = _installation_on(port)
    builds = {
        'Lumascope()': lambda: Lumascope(source_path=root, warn_pre_release=False),
        'create_diagnostic()': lambda: Lumascope.create_diagnostic(source_path=root),
        'create(simulate=False)': lambda: ScopeSession.create(
            complete_settings(), source_path=str(root), warn_pre_release=False
        ),
    }
    before = _scope_threads()
    answered = {}
    with _held_by_another_process(port):
        for name, build in builds.items():
            try:
                build()
            except ScopeInUseError as refused:
                answered[name] = refused.reason
            else:
                answered[name] = 'opened'

    assert answered == dict.fromkeys(builds, 'scope_in_use')
    assert _scope_threads() - before == set()


def test_a_process_drives_one_real_scope(no_hardware):
    first = Lumascope(source_path=installation_of_its_own(), warn_pre_release=False)
    try:
        with pytest.raises(ScopeInUseError):
            Lumascope(source_path=installation_of_its_own(), warn_pre_release=False)
    finally:
        first.disconnect()

    second = Lumascope(source_path=installation_of_its_own(), warn_pre_release=False)
    second.disconnect()


def test_a_real_session_under_a_host_take_opens_and_leaves_the_take_held(no_hardware):
    port = _free_port()
    root = _installation_on(port)
    host = take_instance_lock(root)
    try:
        session = ScopeSession.create(
            complete_settings(), source_path=str(root), warn_pre_release=False
        )
        session.shutdown()
        assert not _another_process_can_take(port)
    finally:
        host.close()
    assert _another_process_can_take(port)


def test_a_host_composes_again_after_a_refused_bring_up_whose_teardown_raised(
    no_hardware, monkeypatch
):
    """The GUI's fallback recompose: the first session's bring-up is
    refused and its teardown's raise is reported, not raised; the scope's
    take was given back all the same, so the second session opens."""
    port = _free_port()
    root = _installation_on(port)
    real_disconnect = lumascope_module.Lumascope._disconnect

    # A stand-in by design: no simulated part raises from its teardown.
    def raising_disconnect(scope):
        real_disconnect(scope)
        raise ScopeDisconnectError({'camera': RuntimeError('did not close')})

    def refused_bring_up(session, replaced):
        raise PermissionError('the bring-up was refused')

    host = take_instance_lock(root)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(lumascope_module.Lumascope, '_disconnect', raising_disconnect)
            patched.setattr(ScopeSession, '_configure_scope', refused_bring_up)
            with pytest.raises(PermissionError):
                ScopeSession.create(
                    complete_settings(), source_path=str(root), warn_pre_release=False
                )
        again = ScopeSession.create(
            complete_settings(), source_path=str(root), warn_pre_release=False
        )
        again.shutdown()
    finally:
        host.close()


def test_a_real_scope_gives_the_lock_back_at_disconnect(no_hardware):
    port = _free_port()
    scope = Lumascope(source_path=_installation_on(port), warn_pre_release=False)
    assert not _another_process_can_take(port)

    scope.disconnect()

    assert _another_process_can_take(port)


def test_a_real_scope_gives_the_lock_back_at_a_disconnect_that_raises(no_hardware, monkeypatch):
    port = _free_port()
    scope = Lumascope(source_path=_installation_on(port), warn_pre_release=False)
    real_disconnect = lumascope_module.Lumascope._disconnect

    # A stand-in by design: no simulated part raises from its teardown.
    def raising_disconnect(scope):
        real_disconnect(scope)
        raise ScopeDisconnectError({'camera': RuntimeError('did not close')})

    monkeypatch.setattr(lumascope_module.Lumascope, '_disconnect', raising_disconnect)
    with pytest.raises(ScopeDisconnectError):
        scope.disconnect()

    assert _another_process_can_take(port)


def test_a_constructor_that_raises_after_its_take_leaves_nothing_running(no_hardware, monkeypatch):
    def missing_timing(scope):
        raise InstallationFileError(scope.source_path / 'camera_timing.json', 'is missing')

    port = _free_port()
    root = _installation_on(port)
    before = _scope_threads()
    monkeypatch.setattr(lumascope_module.Lumascope, '_read_camera_timing', missing_timing)

    with pytest.raises(InstallationFileError):
        Lumascope(source_path=root, warn_pre_release=False)

    assert _scope_threads() - before == set()
    assert _another_process_can_take(port)


def test_a_constructor_that_raises_closes_the_parts_it_opened(monkeypatch):
    def missing_timing(scope):
        raise InstallationFileError(scope.source_path / 'camera_timing.json', 'is missing')

    opened = []

    def keeping(build):
        def kept(*args, **kwargs):
            built = build(*args, **kwargs)
            opened.append(built[0] if isinstance(built, tuple) else built)
            return built

        return kept

    scope_class = lumascope_module.Lumascope
    for builder in ('_build_simulated_motor_board', '_build_simulated_led_board'):
        monkeypatch.setattr(
            scope_class, builder, staticmethod(keeping(getattr(scope_class, builder)))
        )
    monkeypatch.setattr(
        lumascope_module.camera_registry,
        'create',
        keeping(lumascope_module.camera_registry.create),
    )
    monkeypatch.setattr(scope_class, '_read_camera_timing', missing_timing)

    with pytest.raises(InstallationFileError):
        Lumascope(simulate=True, warn_pre_release=False)

    assert [(type(part).__name__, part.is_connected()) for part in opened] == [
        ('SimulatedMotorBoard', False),
        ('SimulatedLEDBoard', False),
        ('SimulatedCamera', False),
    ]


def test_a_take_closed_twice_gives_back_one_take():
    port = _free_port()
    root = _installation_on(port)
    first = take_instance_lock(root)
    second = take_instance_lock(root)

    first.close()
    first.close()

    assert not _another_process_can_take(port)
    second.close()
    assert _another_process_can_take(port)


def test_the_support_reports_retry_builds_its_second_scope(no_hardware, monkeypatch):
    from modules.tech_support_report import FirmwareDiagnostics

    root = installation_of_its_own()
    real_create = Lumascope.create_diagnostic.__func__
    monkeypatch.setattr(
        Lumascope, 'create_diagnostic', classmethod(lambda cls: real_create(cls, root))
    )
    diag = FirmwareDiagnostics()

    diag.connect_standalone()
    first = diag.scope
    diag.connect_standalone()

    try:
        assert (diag.build_failure, diag.scope is not first) == (None, True)
    finally:
        diag.scope.disconnect()


def test_a_simulated_scope_takes_no_lock(no_hardware):
    port = _free_port()
    root = _installation_on(port)
    with _held_by_another_process(port):
        scope = Lumascope(simulate=True, source_path=root, warn_pre_release=False)
        scope.disconnect()


def test_the_server_does_not_start_beside_a_holder(capsys):
    from modules.app_config import get_lvp_lock_port
    from rest.__main__ import main

    with _held_by_another_process(get_lvp_lock_port(get_source_root())):
        assert main(['--simulate']) == 1
    assert 'already running' in capsys.readouterr().err


def test_the_server_takes_no_host():
    from rest.__main__ import main

    with pytest.raises(SystemExit) as refused:
        main(['--host', '0.0.0.0'])
    assert refused.value.code == 2
