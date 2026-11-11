"""A protocol large enough to be worth mentioning still loads.

The step ceiling that used to refuse these files sat BELOW what the app's own
writer produces, so a protocol LumaViewPro had saved could not be reopened.
It also made the size advisory unreachable: the advisory and the refusal shared
the same 10,000-step threshold, and the refusal ran first, so the sentence that
exists to describe a large protocol could never be said about one.

The file ceiling stays, a bound on a load's memory, sized so that a protocol
of the size a customer asked for (a 96-well plate, 4 channels, 200 Z slices:
76,800 steps) is saved and reopened.
"""

import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from _lumascope.modules import common_utils
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.objectives_loader import ObjectiveLoader
from _lumascope.modules.protocol import PROTOCOL_FILE_CEILING_BYTES, Protocol, ProtocolFormatError
from tests.protocol_fixtures import protocol_from_file
from tests.scope_fakes import home_sim_scope
from tests.test_protocol_roundtrip import (
    _build_protocol,
    _make_step,
)
from tests.test_step_label_ssot import _V7_TSV

# Above the advisory threshold (PROTOCOL_SIZE_ADVISORY_STEPS, 10,000) and above
# the step ceiling this suite exists to keep deleted.
LARGE_STEP_COUNT = 12_000

VIDEO_FPS = 10.0


def _large_protocol():
    steps = [_make_step(name=f'A1_BF_{i}', label=f'custom{i:04d}') for i in range(LARGE_STEP_COUNT)]
    return _build_protocol(steps)


def _reload(protocol, tmp_path: pathlib.Path) -> Protocol:
    filepath = tmp_path / 'large_protocol.tsv'
    protocol.to_file(filepath)
    return protocol_from_file(
        file_path=filepath,
    )


@pytest.mark.slow
def test_a_protocol_the_app_wrote_can_be_read_back(tmp_path):
    """The reproduced defect: the writer's own output was unreadable."""
    reloaded = _reload(_large_protocol(), tmp_path)

    assert reloaded.num_steps() == LARGE_STEP_COUNT


@pytest.mark.slow
def test_the_size_advisory_is_reachable_on_a_loaded_protocol(tmp_path):
    """Asserted at the API tier, where the advisory's threshold and sentence live."""
    reloaded = _reload(_large_protocol(), tmp_path)

    advisory = reloaded.size_advisory(video_fps=VIDEO_FPS)

    assert advisory is not None


PEAK_STEPS = 2_000
# The load's peak a step: 3.6 KB when the file's text was held five times
# over and every cell was its own object; 1.7 KB at 2,000 steps once it read
# the file's lines once and held each distinct cell text once.
PEAK_BYTES_A_STEP = 2_000

# The load, measured in a process of its own: tracemalloc counts every
# allocation in its process, so in a test worker it also counts the other
# threads and the garbage earlier tests left (10.5 MB at 2,000 steps in a CI
# worker, against 3.3 MB alone).
_MEASURED_LOAD = """
import gc, sys, tracemalloc
from tests.protocol_fixtures import protocol_from_file
gc.collect()
tracemalloc.start()
loaded = protocol_from_file(file_path=sys.argv[1])
_, peak = tracemalloc.get_traced_memory()
print(loaded.num_steps(), peak)
"""

_CHECKOUT = pathlib.Path(__file__).resolve().parents[1]


def test_a_load_peaks_under_two_kilobytes_a_step(tmp_path):
    """A load's memory at its peak scales with the table it returns, not with copies of the file."""
    steps = [_make_step(name=f'A1_BF_{i}', label=f'custom{i:04d}') for i in range(PEAK_STEPS)]
    path = tmp_path / 'protocol.tsv'
    _build_protocol(steps).to_file(path)

    child = subprocess.run(
        [sys.executable, '-c', _MEASURED_LOAD, str(path)],
        cwd=_CHECKOUT,
        capture_output=True,
        text=True,
    )

    assert child.returncode == 0, child.stderr
    num_steps, peak = map(int, child.stdout.split())
    assert num_steps == PEAK_STEPS
    assert peak / PEAK_STEPS < PEAK_BYTES_A_STEP, f'{peak / PEAK_STEPS:,.0f} bytes a step'


_STACKED_CHANNELS = ('BF', 'Ex405', 'Ex488', 'Ex589')


@pytest.mark.slow
@pytest.mark.parametrize(
    'sim_session',
    [{'zstack': {'step_size': 1.0, 'range': 199.0, 'position': 'center'}}],
    indirect=True,
)
def test_a_protocol_of_the_size_a_customer_asked_for_is_saved_and_reopened(sim_session, tmp_path):
    """96 wells x 4 channels x 200 Z slices, built, saved and loaded through the Session.

    Its file is past the 10 MiB the ceiling once was, so this pins the
    ceiling's size, not only its existence.
    """
    home_sim_scope(sim_session.scope)
    for channel in common_utils.get_channels():
        sim_session.set_channel_acquire(channel, 'image' if channel in _STACKED_CHANNELS else None)
    sim_session.select_labware('96 well microplate')
    built = sim_session.new_protocol(use_zstacking=True)
    assert built.num_steps() == 96 * 4 * 200
    path = tmp_path / 'stacked.tsv'

    sim_session.save_protocol(built, path)
    reloaded = sim_session.load_protocol(path)

    assert path.stat().st_size > 10 * 1024 * 1024
    assert reloaded.num_steps() == 96 * 4 * 200


def test_the_file_ceiling_still_refuses_and_blames_memory_not_corruption(tmp_path):
    """The resource control stays; only its diagnosis changes.

    Size is not evidence of corruption, so a legitimate large file must not be
    told it is damaged.
    """
    huge = tmp_path / 'huge_protocol.tsv'
    huge.write_bytes(b'x' * (PROTOCOL_FILE_CEILING_BYTES + 1))

    with pytest.raises(ProtocolFormatError, match='exhausting memory') as exc_info:
        protocol_from_file(
            file_path=huge,
        )

    assert 'corrupt' not in str(exc_info.value).lower()


def test_a_malformed_file_is_still_refused(tmp_path):
    """Deleting the step ceiling must not make junk loadable."""
    junk = tmp_path / 'junk.tsv'
    junk.write_text('this is not a protocol\n')

    with pytest.raises(ProtocolFormatError, match='Not a valid LumaViewPro Protocol'):
        protocol_from_file(
            file_path=junk,
        )


def refuse_walking_steps_as_rows(monkeypatch) -> None:
    """Make pandas' per-row walks fail the test wherever they are reached.

    A pandas row built per step (``iterrows``, ``apply(axis=1)``) cost a
    69,120-step load about 1.7 of its 2.5 s on the Mac, where the same rows
    read a column at a time cost a tenth of it; on a field PC the load holds
    the UI thread. Patched on the class, so pandas' own use of either is
    caught too.
    """

    def walked(*_args, **_kwargs):
        raise AssertionError('a steps table was walked as one pandas row per step')

    monkeypatch.setattr(pd.DataFrame, 'iterrows', walked)
    monkeypatch.setattr(pd.DataFrame, 'apply', walked)


def test_a_load_and_its_run_check_walk_no_step_as_a_pandas_row(tmp_path, monkeypatch):
    """The load names every step, keys its capture files and parses its video settings
    by column, and the run's check of the loaded steps keys them the same way."""
    path = tmp_path / 'protocol.tsv'
    _build_protocol(
        [
            _make_step(well='A1'),
            _make_step(well='A1'),
            _make_step(well='A2', acquire='video', video_config={'duration': 5}),
        ]
    ).to_file(path)
    refuse_walking_steps_as_rows(monkeypatch)

    loaded = protocol_from_file(file_path=path)
    errors = loaded.validate_for_run(
        objective_helper=ObjectiveLoader(), wellplate_loader=WellPlateLoader(), led_max_ma=1000
    )

    assert list(loaded.steps()['Name']) == ['A1_BF_Z0', 'A1_BF_Z0', 'A2_BF_Z0']
    assert loaded.steps()['Video Config'].iat[2] == {'duration': 5}
    assert [e for e in errors if 'would save captures' in e] == [
        "Steps 1, 2: each would save captures as 'A1_BF_Z0_...'. "
        'Rename these steps so each produces a unique filename.'
    ], errors


def test_an_old_files_label_recovery_walks_no_step_as_a_pandas_row(tmp_path, monkeypatch):
    """A file from before the Label column recovers each step's label by column."""
    path = tmp_path / 'protocol_v7.tsv'
    path.write_text(_V7_TSV)
    refuse_walking_steps_as_rows(monkeypatch)

    loaded = protocol_from_file(file_path=path)

    assert list(loaded.steps()['Label']) == ['', 'Control_preDose', 'Treatment_10x', 'custom0001']
