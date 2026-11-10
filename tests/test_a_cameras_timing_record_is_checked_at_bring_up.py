# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera's timing record is checked at bring-up, and the gate waits at its measured counts.

The records shipped counts with no run behind them -- the dart's lowered the
gate's exposure count from 3 to 2 -- and the gate took them as measured. The
records now carry, per source and direction, a list of measured entries, a
count with the provenance of the run that measured it, naming the microscope
it was measured on; a record of another schema, or an
entry that is no measurement, is refused where the installation is read.
Each source's count is the largest its entries measured, over every
microscope, a move's over every move; a source with none keeps its shipped
count (``SKIP_FRAMES``).
"""

from __future__ import annotations

import json

import pytest

from _lumascope.drivers import camera_profiles
from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.frame_validity import FrameValidity
from _lumascope.modules.path_utils import read_installation_file
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
        FrameValidity.check_timing_record(read_installation_file(path))


def test_the_simulator_reads_a_record_of_its_own():
    """The simulated camera's record is its own: the record it brings up
    with names the simulated camera, not the dart it was built from."""
    simulated = camera_profiles.simulated_profile()
    record = read_installation_file(camera_profiles.timing_record_path(simulated))

    assert record['camera_model'] == simulated.model_name


def test_a_measured_entry_with_its_provenance_is_accepted():
    """A count of 4.0 frames is a whole number of frames, so it is taken."""
    FrameValidity.check_timing_record(
        _record(
            {
                'gain': {'up': [_measured(), _measured(4.0, microscope='LS720')]},
            }
        )
    )


def test_a_record_that_is_no_measurement_is_refused_naming_why():
    no_provenance = _measured()
    del no_provenance['provenance']
    no_microscope = _measured()
    del no_microscope['provenance']['microscope']
    a_delay = {'apply_delay_us': 410.0, 'trials': 40, 'provenance': dict(_PROVENANCE)}
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
        # XY moves the sample: its count is the shipped one, never measured.
        'an XY count': (
            _record({'xy_move': {'out': [_measured()]}}),
            "'xy_move', a move that keeps its shipped count",
        ),
        # A caller's change waits the LED's count, measured once, under led.
        'a caller count': (
            _record({'caller': {'on': [_measured()]}}),
            "'caller', a change that waits the LED's count",
        ),
        # The clock rule a delay was measured for is retired: a record that
        # still holds one is refused by its name, with or without a count.
        'a delay': (
            _record({'exposure': {'up': [a_delay]}}),
            'validity.exposure.up[0] holds apply_delay_us',
        ),
        'a delay beside a count': (
            _record({'gain': {'up': [_measured(apply_delay_us=1.0)]}}),
            'validity.gain.up[0] holds apply_delay_us',
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


def test_a_timing_record_the_reader_refuses_is_refused_at_bring_up(tmp_path, monkeypatch):
    """The record is parsed by the installation's one reader: a key twice is
    refused, as in every installation file, and a record that is not an
    object is the installation's error, not an AttributeError from the check."""
    wrong = {}
    for case, (text, why) in {
        'a key twice': ('{"schema": 3, "schema": 3}', "the key 'schema' twice"),
        'a list': ('[1, 2]', 'holds a list, not a JSON object'),
    }.items():
        record = _bring_up_with(tmp_path, monkeypatch, text)
        try:
            build_scope(simulate=True, sim_model='LS850T').disconnect()
            wrong[case] = 'brought up'
        except InstallationFileError as e:
            if e.file_path != record or why not in str(e):
                wrong[case] = str(e)
        except Exception as e:  # the check met a record it was never given the shape of
            wrong[case] = f'{type(e).__name__}: {e}'

    assert wrong == {}


def test_a_broken_timing_record_is_refused_at_bring_up(tmp_path, monkeypatch):
    broken = _bring_up_with(tmp_path, monkeypatch, '{"skip_frames": ')

    with pytest.raises(InstallationFileError) as refused:
        build_scope(simulate=True, sim_model='LS850T')

    assert refused.value.file_path == broken


def test_a_classic_waits_the_counts_its_camera_was_measured_to_need():
    """Every Classic carries the one FX2 camera, the MT9P031, measured once on
    the LS720: a capture waits that run's counts, with the run behind them.
    Its frame size, measured 2, is not in the record, so it keeps the shipped
    3 (Eric, 2026-11-10: frame size changes are rare)."""
    scope = build_scope(simulate=True, sim_model='LS720')
    try:
        published = scope.capabilities.frame_rules
    finally:
        scope.disconnect()

    measured = {
        source: (rule.count, rule.provenance['microscope'], rule.provenance['run_id'])
        for source, rule in published.items()
        if rule.provenance is not None
    }
    run = '2026-11-10_143549_ronchi_camera_only'
    assert measured == {
        'exposure': (3, 'LS720', run),
        'gain': (3, 'LS720', run),
        'led': (3, 'LS720', run),
        'caller': (3, 'LS720', run),
        'z_move': (1, 'LS720', run),
        'turret': (1, 'LS720', run),
    }
    assert published['frame_size'].count == FrameValidity.SKIP_FRAMES['frame_size']


def test_a_source_its_record_measured_no_count_for_keeps_its_shipped_count():
    """The simulated camera's shipped record measured no count, as the
    shipped cameras' do, so every source's published count is the shipped
    one, with no run behind it."""
    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        published = scope.capabilities.frame_rules
    finally:
        scope.disconnect()

    assert {source: rule.count for source, rule in published.items()} == FrameValidity.SKIP_FRAMES
    assert [source for source, rule in published.items() if rule.provenance is not None] == []


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

    gain = FrameValidity.rules_from_record(record)['gain']

    assert (gain.count, gain.provenance['microscope']) == (4, 'LS720')


def test_a_measured_move_count_is_z_and_the_turrets_and_xy_keeps_its_shipped_count():
    """Z's measured count is the turret's too, since a turret change ends in
    Z's restore. XY keeps its shipped 2 whatever Z measures: it moves the
    sample, whose settling no camera measurement predicts (Eric,
    2026-11-05). A Z of 1 is below XY's 2, so a shared count would lower it."""
    rules = FrameValidity.rules_from_record(_record({'z_move': {'down': [_measured(1)]}}))

    assert (rules['z_move'].count, rules['turret'].count, rules['xy_move'].count) == (1, 1, 2)


def test_a_callers_change_takes_the_leds_rule_shipped_or_measured():
    """A change a caller made outside LVP waits the LED's count: the
    shipped one with no record, and the LED's measured count and run with one."""
    measured = FrameValidity.rules_from_record(_record({'led': {'on': [_measured(5)]}}))
    shipped = FrameValidity.credit_rules()

    assert measured['caller'] == measured['led']
    assert measured['caller'].count == 5
    assert shipped['caller'] == shipped['led']


def test_bring_up_publishes_each_sources_rule_and_a_change_is_judged_by_it(tmp_path, monkeypatch):
    _bring_up_with(tmp_path, monkeypatch, json.dumps(_record({'exposure': {'up': [_measured(4)]}})))

    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        published = scope.capabilities.frame_rules
        scope.camera.set_exposure_ms(scope.camera.read_exposure_ms() + 1.0)
        changes = scope.imaging.recorded_changes()
    finally:
        scope.disconnect()

    assert published['exposure'].count == 4
    assert published['gain'] == FrameValidity.credit_rules()['gain']
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
