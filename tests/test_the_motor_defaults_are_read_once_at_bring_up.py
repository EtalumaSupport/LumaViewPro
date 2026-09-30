# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped motor defaults are read once, from the install, at bring-up.

Every value a motor board does not report itself -- travel limits,
microsteps per mm, ramp parameters -- comes from
`data/motorconfig_defaults.json`. Each driver used to open it relative to
the working directory and run on an empty table when it was not there, so
a scope started from another folder converted positions and bounded moves
by hardcoded fallbacks with only a log line to say so.

Now the bring-up reads it once, through the same resolver as the model
catalogue, and hands it to the motor drivers. A missing or unreadable
file stops the bring-up, naming the file.
"""

import json
import threading

import pytest

import modules.lumascope_api._lumascope as lumascope_module
from drivers.motorconfig import MotorConfig
from modules.scope_session import ScopeSession
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


def _session(model: str = 'LS850T') -> ScopeSession:
    return ScopeSession.create(
        complete_settings(microscope=model),
        simulate=True,
        warn_pre_release=False,
    )


def _defaults_at(monkeypatch, path):
    monkeypatch.setattr(lumascope_module, 'resolve_data_file', lambda *parts: path)


@pytest.mark.parametrize(
    ('content', 'says'),
    [
        (None, 'not found'),
        ('{"Axis Travel Limit": ', 'unreadable'),
        ('[1, 2]', 'not an object'),
    ],
    ids=['missing', 'corrupt', 'not-an-object'],
)
def test_a_bad_defaults_file_stops_the_bring_up_naming_it(monkeypatch, tmp_path, content, says):
    path = tmp_path / 'motorconfig_defaults.json'
    if content is not None:
        path.write_text(content, encoding='utf-8')
    _defaults_at(monkeypatch, path)
    threads_before = set(threading.enumerate())

    with pytest.raises(RuntimeError, match=says) as refused:
        _session()

    assert str(path) in str(refused.value)
    # Refused before anything was started, so nothing is left running.
    assert set(threading.enumerate()) <= threads_before


@pytest.mark.parametrize('model', ['LS620', 'LS850T'])
def test_a_bad_defaults_file_stops_the_bring_up_on_every_model(monkeypatch, tmp_path, model):
    # The motor probe runs on every model, and a board it finds takes the
    # defaults, so a scope with no motors is not exempt.
    _defaults_at(monkeypatch, tmp_path / 'motorconfig_defaults.json')

    with pytest.raises(RuntimeError, match=r'motorconfig_defaults\.json'):
        _session(model)


def test_the_defaults_are_found_from_any_working_directory(monkeypatch, tmp_path):
    # The scope, which builds the motor driver, resolves from the install.
    # (A Session's own data helpers take the source path they are given.)
    monkeypatch.chdir(tmp_path)
    scope = build_scope(
        simulate=True, sim_model='LS850T', warn_pre_release=False, register_atexit=False
    )
    motorconfig = scope._motion_driver.motorconfig
    assert motorconfig.travel_limit_mm('Z') == SHIPPED_MOTOR_DEFAULTS['Axis Travel Limit']['Z']
    assert (
        motorconfig.usteps_per_mm('Z')
        == SHIPPED_MOTOR_DEFAULTS['Axis Microsteps per mm / Objective']['Z']
    )


def test_a_board_merge_leaves_the_shared_defaults_alone():
    defaults = json.loads(json.dumps(SHIPPED_MOTOR_DEFAULTS))
    shipped_z = defaults['Axis Travel Limit']['Z']
    merged = MotorConfig(defaults)
    other = MotorConfig(defaults)

    merged.update_from_board({'Axis Travel Limit': {'Z': shipped_z + 5}})

    assert merged.travel_limit_mm('Z') == shipped_z + 5
    assert other.travel_limit_mm('Z') == shipped_z
    assert defaults['Axis Travel Limit']['Z'] == shipped_z
