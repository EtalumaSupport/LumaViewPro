# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Repair binds a Lumascope Classic to LumaViewPro's driver, through the remedy door.

A Classic Windows has on a driver LumaViewPro cannot use is reported at
bring-up with ``repair_camera_driver`` as its remedy, on an installed Windows
build. ``ScopeSession.repair_camera_driver`` runs the installed driver step as
an administrator, refuses where it cannot run or is not needed, tells a
declined approval and a step that did not bind apart, and on success returns
the new bindings and reports that a restart connects the camera.

The simulated FX2 is the bus, its binding the simulation's fault door. This
host is a Mac, so Windows' approval prompt and the installed build are stood
in; their own run is the Windows bench (5E).
"""

import pytest

from drivers.usb_driver_binding import FX2_FIRMWARE
from modules import fx2_driver_step
from modules.exceptions import (
    CameraDriverRepairFailedError,
    CameraDriverRepairRefusedError,
    CameraNotAvailableError,
)
from modules.path_utils import AppRuntime, launch_root
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


@pytest.fixture
def on_installed_windows(monkeypatch):
    # a stand-in by design: this host is a Mac running from source; Repair
    # needs Windows and an installed build.
    monkeypatch.setattr(fx2_driver_step, 'repair_refusal', lambda: None)


@pytest.fixture
def camera_never_came_up(fx2_session, monkeypatch):
    # a stand-in by design, a simulator gap: the simulated FX2 cannot yet
    # come up refused for its binding (its connect runs at construction and
    # raises out of bring-up), so the scope's camera is read as not up while
    # the device stays on the bus.
    monkeypatch.setattr(type(fx2_session.scope), 'camera_connected', property(lambda s: False))


def _approved_step(session, exit_code=0, binds=True):
    """The step as an administrator approves it: run, bound, exited."""
    runs = []

    def run(executable, argument, wait_s):
        runs.append((executable, argument))
        if binds:
            session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'WinUSB')
        return exit_code

    return run, runs


def test_the_remedy_runs_the_installed_step_and_returns_the_bound_device(
    fx2_session, on_installed_windows, camera_never_came_up, monkeypatch
):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')
    run, runs = _approved_step(fx2_session)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)
    heard = []
    fx2_session.add_outcome_listener(heard.append)

    bindings = fx2_session.apply_remedy(CameraNotAvailableError.repair())

    assert runs == [(launch_root() / 'lvp_fx2_driver_step.exe', 'install')]
    assert [identity.usable for identity in bindings] == [True, True]
    assert 'camera_driver_repaired' in [n.reason for n in heard]


def test_a_declined_approval_is_refused_as_declined(
    fx2_session, on_installed_windows, camera_never_came_up, monkeypatch
):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, None)
    # a stand-in by design: Windows' administrator prompt, declined.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', lambda *a: None)

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'repair_declined'


def test_a_package_windows_did_not_take_is_refused_as_not_accepted(
    fx2_session, on_installed_windows, camera_never_came_up, monkeypatch
):
    # Declined at Windows' question whether to install software from
    # Etaluma, or refused by Windows: the step changed nothing.
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, None)
    run, _ = _approved_step(fx2_session, exit_code=fx2_driver_step.NOT_ACCEPTED_EXIT, binds=False)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'driver_not_accepted'


def test_a_scope_with_no_classic_device_is_refused_not_crashed(on_installed_windows):
    session = ScopeSession.create(
        complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
    )
    try:
        with pytest.raises(CameraDriverRepairRefusedError) as refused:
            session.repair_camera_driver()
    finally:
        session.shutdown()

    assert refused.value.reason == 'no_classic_bindings'


def test_a_usable_binding_is_not_repaired(
    fx2_session, on_installed_windows, camera_never_came_up, monkeypatch
):
    run, runs = _approved_step(fx2_session)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'driver_already_usable'
    assert runs == []


def test_a_camera_that_is_up_is_never_rebound(fx2_session, on_installed_windows, monkeypatch):
    # The running firmware is up on WinUSB; an unbound bootloader identity
    # beside it is another vendor's board, and the step would reinstall the
    # running camera's device.
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')
    run, runs = _approved_step(fx2_session)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'driver_already_usable'
    assert runs == []


@pytest.mark.parametrize(
    'exit_code, binds, said',
    [(1, False, 'exit code 1'), (0, False, 'still on a driver')],
)
def test_a_step_that_did_not_bind_is_a_fault_naming_its_log(
    fx2_session, on_installed_windows, camera_never_came_up, monkeypatch, exit_code, binds, said
):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')
    run, _ = _approved_step(fx2_session, exit_code=exit_code, binds=binds)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)

    with pytest.raises(CameraDriverRepairFailedError) as failed:
        fx2_session.repair_camera_driver()

    assert said in str(failed.value)
    assert str(launch_root() / 'logs') in str(failed.value)


def test_off_windows_repair_is_refused_before_anything_runs(fx2_session):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'repair_needs_windows'


def test_only_an_unusable_driver_offers_repair_and_says_where_it_can_be_installed():
    offered = CameraNotAvailableError('camera_driver_unusable', repairable=True)
    not_here = CameraNotAvailableError('camera_driver_unusable')

    assert offered.remedy.member == 'repair_camera_driver'
    assert 'Install its driver from here' in str(offered)
    assert not_here.remedy is None
    assert 'install a LumaViewPro build that carries it' in str(not_here)
    with pytest.raises(TypeError):
        CameraNotAvailableError('camera_not_detected', repairable=True)


@pytest.fixture
def installed_on_windows(tmp_path, monkeypatch):
    # a stand-in by design: this host is a Mac running from source; the
    # refusal reads the platform, the runtime and the install folder.
    monkeypatch.setattr(fx2_driver_step.sys, 'platform', 'win32')
    monkeypatch.setattr(fx2_driver_step, 'app_runtime', lambda: AppRuntime.INSTALLED)
    monkeypatch.setattr(fx2_driver_step, 'launch_root', lambda: tmp_path)
    return tmp_path


def test_an_installation_without_the_classic_driver_cannot_repair(installed_on_windows):
    assert fx2_driver_step.repair_refusal() == 'repair_needs_driver_package'


def test_an_installation_carrying_the_classic_driver_can_repair(installed_on_windows):
    package = installed_on_windows / fx2_driver_step.PACKAGE_DIR
    package.mkdir(parents=True)
    (package / fx2_driver_step.PACKAGE_INF).write_text('[Version]\n', encoding='ascii')
    (installed_on_windows / fx2_driver_step.DRIVER_STEP_EXE).write_bytes(b'exe')

    assert fx2_driver_step.repair_refusal() is None
