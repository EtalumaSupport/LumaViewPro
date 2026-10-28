# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera's timing record is checked at bring-up, and the gate waits at its measured counts.

The records shipped counts with no run behind them -- the dart's lowered the
gate's exposure count from 3 to 2 -- and the gate took them as measured. The
records now carry, per source and direction, a list of measured entries, a
delay or a count with the provenance of the run that measured it, a count's
naming the microscope it was measured on; a record of another schema, or an
entry that is no measurement, is refused where the installation is read.
Each source's count is the largest its entries measured, over every
microscope, a move's over every move; a source with none keeps its shipped
count (``SKIP_FRAMES``).
"""

from __future__ import annotations

import json

import pytest

from drivers import camera_profiles
from modules.exceptions import InstallationFileError
from modules.frame_validity import FrameValidity, Rule
from tests.scope_fakes import bind_settings_like_a_session, build_scope

_PROVENANCE = {
    'run_id': 'run-1',
    'host': 'bench-pc',
    'os': 'Windows 11',
    'lvp_version': '4.0.0',
    'plugin_version': '1.0.66',
    'measured_date': '2026-10-16',
}


def _measured(max_frames=2, microscope='LS850T', **overrides):
    value = {
        'delivered_frames': {'p50': 1, 'p95': max_frames, 'max': max_frames},
        'trials': 40,
        'provenance': _PROVENANCE | {'microscope': microscope},
    }
    value.update(overrides)
    return value


def _delay(apply_delay_us):
    return {'apply_delay_us': apply_delay_us, 'trials': 40, 'provenance': dict(_PROVENANCE)}


def _record(validity):
    return {'schema': 3, 'camera_model': 'daA3840-45um', 'validity': validity}


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


def test_a_measured_entry_with_its_provenance_is_accepted():
    """A count of 4.0 frames is a whole number of frames, so it is taken."""
    FrameValidity.check_timing_record(
        _record(
            {
                'gain': {'up': [_measured(), _measured(4.0, microscope='LS720')]},
                'exposure': {'long_to_short': [_delay(410.0)]},
            }
        )
    )


def test_a_record_that_is_no_measurement_is_refused_naming_why():
    no_provenance = _measured()
    del no_provenance['provenance']
    no_microscope = _measured()
    del no_microscope['provenance']['microscope']
    both = _measured(apply_delay_us=1.0)
    no_measure = {'trials': 1, 'provenance': dict(_PROVENANCE)}
    refused = {
        'the old schema': ({'skip_frames': {'gain': 3}}, 'schema None'),
        'schema 2': (
            {'schema': 2, 'validity': {'gain': {'up': _measured()}}},
            'schema 2, not 3',
        ),
        'no validity group': ({'schema': 3}, 'no validity group'),
        'a source that is no group': (_record({'gain': [1, 2]}), 'not a group of directions'),
        'a direction that is one value': (
            _record({'gain': {'up': _measured()}}),
            'validity.gain.up is not a list of measured entries',
        ),
        'a direction with no entries': (
            _record({'gain': {'up': []}}),
            'validity.gain.up is not a list of measured entries',
        ),
        'an entry that is no object': (_record({'gain': {'up': [3]}}), 'not a measured value'),
        'an unknown source': (
            _record({'focus_wobble': {'up': [_measured()]}}),
            "'focus_wobble'",
        ),
        'no measure': (_record({'gain': {'up': [no_measure]}}), 'no measure'),
        'both measures': (
            _record({'gain': {'up': [both]}}),
            "['apply_delay_us', 'delivered_frames']",
        ),
        'a spread without max': (
            _record({'gain': {'up': [_measured(delivered_frames={'p50': 1, 'p95': 2})]}}),
            'p50, p95 and max',
        ),
        'a negative count': (
            _record(
                {'gain': {'up': [_measured(delivered_frames={'p50': -1, 'p95': 2, 'max': 2})]}}
            ),
            'each 0 or more',
        ),
        'a count that is not whole': (
            _record({'gain': {'up': [_measured(2.5)]}}),
            'validity.gain.up[0].delivered_frames.max is 2.5, not a whole number of frames',
        ),
        # A delay is the least first-new start over the trials, one bound; a
        # spread is the old latest-stale measure, a bound from below.
        'a delay given as a spread': (
            _record({'exposure': {'up': [_delay({'p50': 1, 'p95': 2, 'max': 2})]}}),
            'apply_delay_us is not one number',
        ),
        'a negative delay': (
            _record({'exposure': {'up': [_delay(-5.0)]}}),
            'apply_delay_us is not one number, 0 or more',
        ),
        'no trials': (_record({'gain': {'up': [_measured(trials=0)]}}), 'no trials'),
        'no provenance': (_record({'gain': {'up': [no_provenance]}}), 'has no provenance'),
        'a provenance without its run': (
            _record({'gain': {'up': [_measured(provenance=_PROVENANCE | {'run_id': ''})]}}),
            "['run_id', 'microscope']",
        ),
        # A count is the camera's on one microscope: one that names none
        # cannot be kept beside another microscope's.
        'a count without its microscope': (
            _record({'gain': {'up': [_measured(), no_microscope]}}),
            "validity.gain.up[1] has no provenance for ['microscope']",
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


def test_a_source_its_record_measured_no_count_for_keeps_its_shipped_count():
    """The simulated camera's shipped record measured delays and no count, so
    every source's published count is the shipped one, and a credit rule's
    has no run behind it."""
    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        published = scope.capabilities.frame_rules
    finally:
        scope.disconnect()

    assert {source: rule.count for source, rule in published.items()} == FrameValidity.SKIP_FRAMES
    assert [
        source
        for source, rule in published.items()
        if rule.rule is Rule.CREDIT and rule.provenance is not None
    ] == []


def _delays(**by_direction):
    """A source's directions, each one delay with its own run."""
    return {
        direction: [_delay(us) | {'provenance': _PROVENANCE | {'run_id': f'run-{direction}'}}]
        for direction, us in by_direction.items()
    }


def test_a_latched_cameras_moves_run_the_clock_rule_and_its_other_sources_need_a_delay():
    """On a camera with a clock latch and no move count measured, every move
    runs the clock rule: at the delay its record measured, the slower
    direction's with the run that measured it, or at the stop when the record
    has none or there is no record. Another source runs it only with a
    measured delay. A camera without a latch runs the credit rule for
    everything."""
    record = _record(
        {
            'exposure': _delays(up=300.0, down=500.0),
            'z_move': _delays(out=70.0, back=90.0),
            'gain': {'up': [_measured()]},
        }
    )

    latched = FrameValidity.rules_from_record(record, has_clock_latch=True)
    no_record = FrameValidity.rules_from_record(None, has_clock_latch=True)
    unlatched = FrameValidity.rules_from_record(record, has_clock_latch=False)

    def measured(rules):
        return {
            source: (rule.delay_us, rule.provenance.get('run_id'))
            for source, rule in rules.items()
            if rule.rule is Rule.CLOCK
        }

    assert measured(latched) == {
        'exposure': (500.0, 'run-down'),
        'z_move': (90.0, 'run-back'),
        'xy_move': (0.0, None),
        'turret': (0.0, None),
    }
    assert measured(no_record) == dict.fromkeys(FrameValidity.MOTION_SOURCES, (0.0, None))
    assert set(latched) == set(no_record) == set(FrameValidity.SKIP_FRAMES)
    assert {rule.rule for rule in unlatched.values()} == {Rule.CREDIT}


def test_the_largest_count_over_the_microscopes_is_taken_with_its_microscope():
    """A camera measured on two microscopes waits the larger count, whichever
    direction or microscope measured it, and publishes the run and the
    microscope behind it."""
    record = _record(
        {
            'gain': {
                'up': [_measured(2, microscope='LS850T'), _measured(4, microscope='LS720')],
                'down': [_measured(3, microscope='LS850T')],
            }
        }
    )

    gain = FrameValidity.rules_from_record(record, has_clock_latch=False)['gain']

    assert (gain.rule, gain.count, gain.provenance['microscope']) == (Rule.CREDIT, 4, 'LS720')


def test_a_measured_move_count_is_every_moves_and_outranks_the_stop_rule():
    """A move's count is the camera's whatever the axis: an ``xy_move`` entry
    sets the count of all three moves, and on a camera with a clock latch it
    replaces the rule at the stop."""
    record = _record({'xy_move': {'out': [_measured(5)]}, 'z_move': {'up': [_measured(3)]}})

    rules = FrameValidity.rules_from_record(record, has_clock_latch=True)

    moves = {source: (rules[source].rule, rules[source].count) for source in rules}
    assert {source: moves[source] for source in FrameValidity.MOTION_SOURCES} == dict.fromkeys(
        FrameValidity.MOTION_SOURCES, (Rule.CREDIT, 5)
    )


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


def test_a_measured_count_makes_a_capture_wait_for_the_frame_after_it(tmp_path, monkeypatch):
    """A gain count measured at 4 is what a capture waits: the frame it
    returns is at least the fifth handed over after the change."""
    _bring_up_with(tmp_path, monkeypatch, json.dumps(_record({'gain': {'up': [_measured(4)]}})))

    scope = build_scope(simulate=True, sim_model='LS850T')
    bind_settings_like_a_session(scope)
    scope._camera_driver.set_timing_mode('fast')
    scope.imaging.enable_streaming()
    try:
        scope.camera.set_gain_db(scope.camera.read_gain_db() + 1.0)
        record = scope.imaging.capture_still(accept_dark=True).record
    finally:
        scope.imaging.disable_streaming()
        scope.disconnect()

    gain = next(j for j in record.judgments if j.change.source == 'gain')
    assert (gain.change.skip, gain.counts) == (4, True)
    assert gain.frames_after >= 5
