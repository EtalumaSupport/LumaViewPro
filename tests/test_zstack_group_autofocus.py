# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack is focused once and placed with the found focus on its reference slice.

The contract (``Protocol.apply_zstacking``, ``focus_sweep_step``,
``apply_zstack_group_focus``): a z-stack group at one position is focused
by one sweep, in the light of a channel the user asked to focus and centred
on that channel's reference slice; the found focus lands on that reference
slice -- the slice built at the user's focus: the middle for a centre
stack (the lower middle for an even count), the top for top, the bottom for
bottom -- and every slice sits at its built distance from it, the
``Z-Stack Offset`` each slice records. This holds in a run and in Autofocus
All Steps, whatever wrote the steps' autofocus settings.

Bug history: the reference had no store. ``apply_zstacking`` marked it by
setting Auto_Focus on that one slice, and every other writer of Auto_Focus
erased the mark; Autofocus All Steps turned every slice's on, so the first
flagged slice -- the floor -- took the found focus, and the scan wrote that
ladder into the user's protocol (field, 2026-09-11: ``AF DONE
best=6380.91um``, slices at 6380.9 / 6580.9 / 6780.9 on a centre stack).
And the sweep, fired at the stack's first slice, centred where the stage
stood, the floor, not on the reference slice: a top stack's sweep could
miss the user's focus entirely (CI, 2026-11-02: ``AF START center=4900.0
z=[4825.0, 4975.0]`` for a reference at 5000).

The runs are real scans on the simulated scope, whose sample comes into
focus at ``FOCAL_Z_UM``, away from the layer focus the stack is built
around; each slice's Z is read from the file it saved.
"""

from __future__ import annotations

import pathlib

import pytest

from _lumascope.modules.coord_transformations import AxisLimits
from _lumascope.modules.image_utils import read_file_record
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.objectives_loader import ObjectiveLoader
from _lumascope.modules.protocol import StepEditRefusedError
from _lumascope.modules.protocol_time_estimator import ProtocolTimeEstimator
from _lumascope.modules.scope_session import ScopeSession
from tests.protocol_fixtures import protocol_from_file
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0
from tests.test_protocol_roundtrip import _build_protocol, _make_step

_WIDE_Z = {'Z': AxisLimits(min_um=0.0, max_um=100_000.0)}
LAYER_FOCUS_UM = 5000.0
# Inside the 10x objective's first sweep (75 um either side) centred on the
# reference slice, which a stack is built with at the layer focus, and far
# enough from it that a stack left where it was built is told apart.
FOCAL_Z_UM = 5030.0
STEP_UM = 50.0
# The 10x objective's final pass steps 8 um.
FOUND_WITHIN_UM = 8.0
# The simulated stage reports a Z a few nm from the one it was sent to.
STAGE_WITHIN_UM = 0.01
WAIT_S = 120.0
CENTRE = [-50.0, 0.0, 50.0]


def _built(reference: str, range_um: float, auto_focus: bool):
    """One BF step at the layer focus, stacked: a real Protocol, no scope."""
    protocol = _build_protocol(
        [_make_step(z=LAYER_FOCUS_UM, z_slice=-1, zstack_group_id=-1, auto_focus=auto_focus)]
    )
    protocol.apply_zstacking(
        zstack_params={'range': range_um, 'step_size': STEP_UM, 'z_reference': reference},
        axis_limits=_WIDE_Z,
    )
    return protocol


# --- The stack as built --------------------------------------------------


def test_every_slice_records_its_offset_from_its_reference_slice():
    """Built from a step with autofocus on: each slice's offset, and the flag on all."""
    wrong = []
    for reference, range_um, offsets in (
        ('center', 100.0, CENTRE),
        ('top', 100.0, [-100.0, -50.0, 0.0]),
        ('bottom', 100.0, [0.0, 50.0, 100.0]),
        # No slice is at the focus; the lower middle is the reference.
        ('center', 150.0, [-50.0, 0.0, 50.0, 100.0]),
    ):
        steps = _built(reference, range_um, auto_focus=True).steps()
        got = list(steps['Z-Stack Offset']), list(steps['Auto_Focus'])
        if got != (offsets, [True] * len(offsets)):
            wrong.append(f'{reference} {range_um}: {got}, expected offsets {offsets}, all focused')
    assert not wrong, '\n'.join(wrong)


# --- The run --------------------------------------------------------------


@pytest.fixture
def focusing_session():
    """A session on the simulated scope whose sample is in focus at FOCAL_Z_UM."""
    settings = complete_settings(objective_id='10x Oly')
    settings['protocol']['labware'] = 'Center Plate'
    session = ScopeSession.create(settings, simulate=True)
    home_sim_scope(session.scope)
    session.scope._camera_driver.set_test_pattern(True, 'focus_target')
    session.scope.imaging.enable_streaming()
    session.scope._camera_driver.set_focal_z(FOCAL_Z_UM)
    session.set_channel_acquire('BF', 'image')
    session.save_channel_focus('BF', LAYER_FOCUS_UM)
    session.update_settings('BF.autofocus', True)
    session.update_settings('zstack.step_size', STEP_UM)
    yield session
    session.shutdown()


def _stack(session, *, reference='center', range_um=100.0):
    session.update_settings('zstack.position', reference)
    session.update_settings('zstack.range', range_um)
    return session.new_protocol(use_zstacking=True)


def _captured_z(session, protocol, run_dir: pathlib.Path) -> list[float]:
    """Each slice's Z, lowest first, as the files a scan of ``protocol`` saved state it."""
    run = session.protocol_runner().start_single_scan(
        protocol, parent_dir=run_dir, enable_image_saving=True
    )
    outcome = run.result(timeout=WAIT_S)
    assert outcome.status == 'completed', outcome
    # The outcome settles once the run lets go of the scope; its files once
    # they are written.
    run.files.result(timeout=WAIT_S)
    return sorted(
        read_file_record(path)['z_pos_um'] for path in run_dir.rglob('*.tif*') if path.is_file()
    )


def _off_the_ladder(z: list[float], offsets: list[float]) -> str | None:
    """Why ``z`` is not the ladder of ``offsets`` with its reference at the found focus; None if it is."""
    reference = z[offsets.index(0.0)]
    if abs(reference - FOCAL_Z_UM) > FOUND_WITHIN_UM:
        return f'the reference slice is at {reference}, not the focus near {FOCAL_Z_UM}: {z}'
    if [value - reference for value in z] != pytest.approx(offsets, abs=STAGE_WITHIN_UM):
        return f'the slices {z} are not {offsets} from the reference slice'
    return None


@pytest.mark.slow
@pytest.mark.parametrize(
    # parametrize-justified: each reference is a scan of its own.
    ('reference', 'range_um', 'offsets'),
    [
        ('center', 100.0, CENTRE),
        # The first slice 300 um below the reference: a sweep centred on it
        # reaches 225 um short of the sample's focus.
        ('top', 300.0, [-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0]),
        ('bottom', 100.0, [0.0, 50.0, 100.0]),
        ('center', 150.0, [-50.0, 0.0, 50.0, 100.0]),
    ],
)
def test_a_scan_places_the_found_focus_on_the_reference_slice(
    focusing_session, tmp_path, reference, range_um, offsets
):
    """The sweep is centred on the reference slice, wherever the stack's first slice is."""
    protocol = _stack(focusing_session, reference=reference, range_um=range_um)
    z = _captured_z(focusing_session, protocol, tmp_path)
    assert _off_the_ladder(z, offsets) is None


@pytest.mark.slow
def test_autofocus_all_steps_writes_the_ladder_into_the_callers_protocol(focusing_session):
    """The field case: the scan turns every slice's autofocus on in its own copy."""
    focusing_session.update_settings('BF.autofocus', False)
    protocol = _stack(focusing_session)
    outcome = (
        focusing_session.protocol_runner()
        .start_autofocus_all_steps(protocol)
        .result(timeout=WAIT_S)
    )
    assert outcome.status == 'completed', outcome
    assert outcome.focus_written is True, outcome
    assert _off_the_ladder(sorted(protocol.steps()['Z']), CENTRE) is None


def _every_step_on(session, protocol):
    session.set_step_autofocus(protocol, enabled=False)
    session.set_step_autofocus(protocol, enabled=True)


def _the_floor_on(session, protocol):
    session.set_step_autofocus(protocol, enabled=False)
    session.set_step_autofocus(protocol, enabled=True, step_index=0)


def _the_top_moved_by_hand(session, protocol):
    session.scope.protocols.set_step_z(protocol, 2, LAYER_FOCUS_UM + 7.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    # parametrize-justified: each edit is a scan of its own.
    'edit',
    [_every_step_on, _the_floor_on, _the_top_moved_by_hand],
)
def test_the_ladder_survives_each_writer_of_the_stacks_steps(focusing_session, tmp_path, edit):
    protocol = _stack(focusing_session)
    edit(focusing_session, protocol)
    assert _off_the_ladder(_captured_z(focusing_session, protocol, tmp_path), CENTRE) is None


@pytest.mark.slow
def test_a_stack_whose_reference_slice_was_deleted_is_placed_from_where_it_was(
    focusing_session, tmp_path
):
    protocol = _stack(focusing_session)
    focusing_session.delete_step(protocol, 1)
    low, high = _captured_z(focusing_session, protocol, tmp_path)
    assert high - low == pytest.approx(100.0, abs=STAGE_WITHIN_UM), (low, high)
    assert abs((low + high) / 2 - FOCAL_Z_UM) <= FOUND_WITHIN_UM, (low, high)


@pytest.mark.slow
def test_a_stack_whose_autofocus_is_turned_off_stays_where_it_was_built(focusing_session, tmp_path):
    protocol = _stack(focusing_session)
    focusing_session.set_step_autofocus(protocol, enabled=False, step_index=1)
    assert list(protocol.steps()['Auto_Focus']) == [False, False, False]
    assert _captured_z(focusing_session, protocol, tmp_path) == pytest.approx(
        [4950.0, 5000.0, 5050.0], abs=STAGE_WITHIN_UM
    )


def test_a_slice_of_a_stack_keeps_its_channel(focusing_session):
    protocol = _stack(focusing_session)
    before = protocol.steps()
    session = focusing_session
    session.set_channel_acquire(LED0, 'image')
    session.scope.motion.move_z(protocol.step_row(step_index=0)['Z'])
    with pytest.raises(StepEditRefusedError, match='slice of a z-stack of BF'):
        session.update_step(protocol, 0, channel=LED0)
    assert protocol.steps().equals(before)


def test_focusing_a_channel_places_its_stacks_with_their_reference_at_the_focus(
    focusing_session,
):
    protocol = _stack(focusing_session)
    focusing_session.scope.motion.move_z(5100.0)
    assert focusing_session.apply_focus_to_channel_steps(protocol, 'BF') == 3
    assert list(protocol.steps()['Z']) == pytest.approx(
        [5050.0, 5100.0, 5150.0], abs=STAGE_WITHIN_UM
    )


@pytest.mark.slow
def test_a_groups_sweep_runs_in_its_reference_slices_light(focusing_session, tmp_path):
    """A group holds every channel at its position in Z order, so its first
    slice -- the one that fires the sweep -- can be another channel's. The
    sweep runs in the reference slice's light all the same: LED 0 saved 10 um
    below BF sorts first here, and BF is the channel that autofocuses."""
    session = focusing_session
    session.set_channel_acquire(LED0, 'image')
    session.save_channel_focus(LED0, LAYER_FOCUS_UM - 10.0)
    protocol = _stack(session)
    first = protocol.step_row(step_index=0)
    reference = protocol.step_row(step_index=protocol.focus_sweep_step(step_index=0))
    assert (first['Color'], reference['Color']) == (LED0, 'BF')

    lit = []

    def on_led(address, name, state, illumination_ma):
        if state is LedState.LIT:
            lit.append((name, illumination_ma))

    session.scope.illumination.add_led_listener(on_led)
    outcome = (
        session.protocol_runner()
        .start_single_scan(protocol, parent_dir=tmp_path, enable_image_saving=False)
        .result(timeout=WAIT_S)
    )

    assert outcome.status == 'completed', outcome
    # Nothing is lit before the first slice's sweep, so the first light is the sweep's.
    assert lit[0] == ('BF', reference['Illumination'])


# --- A file from before the offset was recorded --------------------------


def _as_v9_file(protocol, path: pathlib.Path, flags: list[bool]) -> pathlib.Path:
    """``protocol`` saved as a v9 build saved it: no Z-Stack Offset, each slice flagged as given."""
    protocol.to_file(path)
    lines = path.read_text(encoding='utf-8').splitlines()
    header_at = lines.index('Steps') + 1
    header = lines[header_at].split('\t')
    dropped, flag_at = header.index('Z-Stack Offset'), header.index('Auto_Focus')
    out = []
    for number, line in enumerate(lines):
        cells = line.split('\t')
        if cells[0] == 'Version':
            cells = ['Version', '9']
        elif number >= header_at:
            if number > header_at:
                cells[flag_at] = str(flags[number - header_at - 1])
            del cells[dropped]
        out.append('\t'.join(cells))
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return path


def test_an_old_files_stack_takes_its_reference_from_its_flags(tmp_path):
    """One flagged slice (the v9 builds) is the reference; with none or every
    one flagged (3.x and early betas), the middle slice is."""
    protocol = _built('top', 100.0, auto_focus=False)
    wrong = []
    for flags, offsets, focused in (
        ([False, False, True], [-100.0, -50.0, 0.0], True),
        ([True, True, True], CENTRE, True),
        ([False, False, False], CENTRE, False),
    ):
        steps = protocol_from_file(_as_v9_file(protocol, tmp_path / 'v9.tsv', flags)).steps()
        got = list(steps['Z-Stack Offset']), list(steps['Auto_Focus'])
        if got != (offsets, [focused] * 3):
            wrong.append(f'{flags}: {got}, expected {offsets} and every slice {focused}')
    assert not wrong, '\n'.join(wrong)


# --- The estimate --------------------------------------------------------


def test_the_estimate_counts_one_sweep_per_stack():
    estimate = ProtocolTimeEstimator(ObjectiveLoader()).estimate(
        _built('center', 100.0, auto_focus=True)
    )
    swept = [step.autofocus_time_s > 0 for step in estimate.scan_estimate.step_estimates]
    assert swept == [True, False, False]
