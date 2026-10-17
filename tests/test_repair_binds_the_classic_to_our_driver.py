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
from modules.path_utils import launch_root
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
    fx2_session, on_installed_windows, monkeypatch
):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, 'libusbK')
    run, runs = _approved_step(fx2_session)
    # a stand-in by design: Windows' administrator prompt, absent off Windows.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', run)
    heard = []
    fx2_session.add_outcome_listener(heard.append)

    bindings = fx2_session.apply_remedy(CameraNotAvailableError.REPAIR)

    assert runs == [(launch_root() / 'lvp_fx2_driver_step.exe', 'install')]
    assert [identity.usable for identity in bindings] == [True, True]
    assert 'camera_driver_repaired' in [n.reason for n in heard]


def test_a_declined_approval_is_refused_as_declined(fx2_session, on_installed_windows, monkeypatch):
    fx2_session.scope.simulation.bind_camera_usb_driver(FX2_FIRMWARE, None)
    # a stand-in by design: Windows' administrator prompt, declined.
    monkeypatch.setattr(fx2_driver_step, 'run_as_administrator', lambda *a: None)

    with pytest.raises(CameraDriverRepairRefusedError) as refused:
        fx2_session.repair_camera_driver()

    assert refused.value.reason == 'repair_declined'


def test_a_usable_binding_is_not_repaired(fx2_session, on_installed_windows, monkeypatch):
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
    fx2_session, on_installed_windows, monkeypatch, exit_code, binds, said
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


def test_only_an_unusable_driver_offers_repair():
    assert (
        CameraNotAvailableError('camera_driver_unusable', repairable=True).remedy.member
        == 'repair_camera_driver'
    )
    assert CameraNotAvailableError('camera_driver_unusable').remedy is None
    with pytest.raises(TypeError):
        CameraNotAvailableError('camera_not_detected', repairable=True)
