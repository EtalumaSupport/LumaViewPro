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

import gc
import pathlib
import tracemalloc

import pytest

from _lumascope.modules import common_utils
from _lumascope.modules.protocol import PROTOCOL_FILE_CEILING_BYTES, Protocol, ProtocolFormatError
from tests.protocol_fixtures import protocol_from_file
from tests.scope_fakes import home_sim_scope
from tests.test_protocol_roundtrip import (
    _build_protocol,
    _make_step,
)

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
# over and every cell was its own object; 1.3 KB measured once it read the
# file's lines once and held each distinct cell text once.
PEAK_BYTES_A_STEP = 2_000


def test_a_load_peaks_under_two_kilobytes_a_step(tmp_path):
    """A load's memory at its peak scales with the table it returns, not with copies of the file."""
    steps = [_make_step(name=f'A1_BF_{i}', label=f'custom{i:04d}') for i in range(PEAK_STEPS)]
    path = tmp_path / 'protocol.tsv'
    _build_protocol(steps).to_file(path)
    gc.collect()

    tracemalloc.start()
    try:
        loaded = protocol_from_file(file_path=path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert loaded.num_steps() == PEAK_STEPS
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
