# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera's timing record is checked at bring-up, and the gate drains at its own counts.

The records shipped counts with no run behind them -- the dart's lowered the
gate's exposure count from 3 to 2 -- and the gate took them as measured. The
records now carry, per source and direction, a measured delay or count with
the provenance of the run that measured it; a record of another schema, or a
value that is no measurement, is refused where the installation is read.
The 2026-05 counts, with no surviving run, are dropped (Eric, 2026-10-16),
so the gate drains at its own counts until a measured value replaces them.
"""

from __future__ import annotations

import json

import pytest

from drivers import camera_profiles
from modules.exceptions import InstallationFileError
from modules.frame_validity import FrameValidity, Rule
from tests.scope_fakes import build_scope

_PROVENANCE = {
    'run_id': 'run-1',
    'host': 'bench-pc',
    'os': 'Windows 11',
    'lvp_version': '4.0.0',
    'plugin_version': '1.0.66',
    'measured_date': '2026-10-16',
}


def _measured(**overrides):
    value = {
        'delivered_frames': {'p50': 1, 'p95': 2, 'max': 2},
        'trials': 40,
        'provenance': dict(_PROVENANCE),
    }
    value.update(overrides)
    return value


def _delay(apply_delay_us):
    return {'apply_delay_us': apply_delay_us, 'trials': 40, 'provenance': dict(_PROVENANCE)}


def _record(validity):
    return {'schema': 2, 'camera_model': 'daA3840-45um', 'validity': validity}


def _bring_up_with(tmp_path, monkeypatch, text):
    record = tmp_path / 'daA3840-45um.json'
    record.write_text(text)
    # The record's location is the installation's; pointing it at a test's
    # file is the one seam that does not write into the data folder.
    monkeypatch.setattr(camera_profiles, 'timing_record_path', lambda profile: record)
    return record


def test_every_shipped_record_is_one_this_release_reads():
    folder = camera_profiles.timing_record_path(camera_profiles.simulated_profile()).parent
    records = sorted(folder.glob('*.json'))

    assert records
    for path in records:
        FrameValidity.check_timing_record(json.loads(path.read_text()))


def test_the_simulator_reads_a_record_of_its_own():
    """The simulated sensor's delays are its own: the record it brings up
    with names the simulated camera, not the dart it was built from."""
    simulated = camera_profiles.simulated_profile()

    assert camera_profiles.timing_record(simulated)['camera_model'] == simulated.model_name


def test_a_measured_value_with_its_provenance_is_accepted():
    FrameValidity.check_timing_record(
        _record({'gain': {'up': _measured()}, 'exposure': {'long_to_short': _delay(410.0)}})
    )


def test_a_record_that_is_no_measurement_is_refused_naming_why():
    no_provenance = _measured()
    del no_provenance['provenance']
    both = _measured(apply_delay_us=1.0)
    no_measure = {'trials': 1, 'provenance': dict(_PROVENANCE)}
    refused = {
        'the old schema': ({'skip_frames': {'gain': 3}}, 'schema None'),
        'no validity group': ({'schema': 2}, 'no validity group'),
        'a source that is no group': (_record({'gain': [1, 2]}), 'not a group of directions'),
        'a value that is no object': (_record({'gain': {'up': 3}}), 'not a measured value'),
        'an unknown source': (_record({'focus_wobble': {'up': _measured()}}), "'focus_wobble'"),
        'no measure': (_record({'gain': {'up': no_measure}}), 'no measure'),
        'both measures': (
            _record({'gain': {'up': both}}),
            "['apply_delay_us', 'delivered_frames']",
        ),
        'a spread without max': (
            _record({'gain': {'up': _measured(delivered_frames={'p50': 1, 'p95': 2})}}),
            'p50, p95 and max',
        ),
        'a negative count': (
            _record({'gain': {'up': _measured(delivered_frames={'p50': -1, 'p95': 2, 'max': 2})}}),
            'each 0 or more',
        ),
        # A delay is the least first-new start over the trials, one bound; a
        # spread is the old latest-stale measure, a bound from below.
        'a delay given as a spread': (
            _record({'exposure': {'up': _delay({'p50': 1, 'p95': 2, 'max': 2})}}),
            'apply_delay_us is not one number',
        ),
        'a negative delay': (
            _record({'exposure': {'up': _delay(-5.0)}}),
            'apply_delay_us is not one number, 0 or more',
        ),
        'no trials': (_record({'gain': {'up': _measured(trials=0)}}), 'no trials'),
        'no provenance': (_record({'gain': {'up': no_provenance}}), 'has no provenance'),
        'a provenance without its run': (
            _record({'gain': {'up': _measured(provenance=_PROVENANCE | {'run_id': ''})}}),
            "['run_id']",
        ),
    }
    wrong = {}
    for name, (record, why) in refused.items():
        try:
            FrameValidity.check_timing_record(record)
            wrong[name] = 'accepted'
        except ValueError as e:
            if why not in str(e):
                wrong[name] = str(e)

    assert wrong == {}


def test_a_record_of_the_old_schema_is_refused_at_bring_up(tmp_path, monkeypatch):
    record = _bring_up_with(tmp_path, monkeypatch, json.dumps({'skip_frames': {'gain': 3}}))

    with pytest.raises(InstallationFileError) as refused:
        build_scope(simulate=True, sim_model='LS850T')

    assert refused.value.file_path == record


def test_a_broken_timing_record_is_refused_at_bring_up(tmp_path, monkeypatch):
    broken = _bring_up_with(tmp_path, monkeypatch, '{"skip_frames": ')

    with pytest.raises(InstallationFileError) as refused:
        build_scope(simulate=True, sim_model='LS850T')

    assert refused.value.file_path == broken


def test_the_simulated_dart_drains_at_the_gates_own_counts():
    """The simulated camera's shipped record holds no measured value, so it
    drains at the gate's own counts, as the dart's once did not: its 2026-05
    counts (gain 3, exposure 2) were dropped."""
    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        assert scope.imaging.frame_validity.SKIP_FRAMES == FrameValidity.SKIP_FRAMES
    finally:
        scope.disconnect()


def _delays(**by_direction):
    """A source's directions, each a delay with its own run."""
    return {
        direction: _delay(us) | {'provenance': _PROVENANCE | {'run_id': f'run-{direction}'}}
        for direction, us in by_direction.items()
    }


def test_a_source_runs_the_clock_rule_only_with_a_latch_and_a_measured_delay():
    """The slower direction sets the delay, with the run that measured it: a
    capture waits one delay per source whichever way the change went."""
    record = _record({'exposure': _delays(up=300.0, down=500.0), 'gain': {'up': _measured()}})

    latched = FrameValidity.rules_from_record(record, has_clock_latch=True)
    unlatched = FrameValidity.rules_from_record(record, has_clock_latch=False)

    exposure = latched['exposure']
    assert (exposure.rule, exposure.delay_us, exposure.provenance['run_id']) == (
        Rule.CLOCK,
        500.0,
        'run-down',
    )
    assert {s for s, rule in latched.items() if rule.rule is Rule.CLOCK} == {'exposure'}
    assert set(latched) == set(FrameValidity.SKIP_FRAMES)
    assert {rule.rule for rule in unlatched.values()} == {Rule.CREDIT}


def test_bring_up_publishes_each_sources_rule_and_a_change_is_judged_by_it(tmp_path, monkeypatch):
    _bring_up_with(tmp_path, monkeypatch, json.dumps(_record({'exposure': _delays(up=500.0)})))

    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        published = scope.capabilities.frame_rules
        scope.camera.set_exposure_ms(scope.camera.read_exposure_ms() + 1.0)
        changes = scope.imaging.recorded_changes()
    finally:
        scope.disconnect()

    assert (published['exposure'].rule, published['exposure'].delay_us) == (Rule.CLOCK, 500.0)
    assert published['gain'].rule is Rule.CREDIT
    assert changes['exposure'].rule == published['exposure']
