"""A protocol large enough to be worth mentioning still loads.

The step ceiling that used to refuse these files sat BELOW what the app's own
writer produces, so a protocol LumaViewPro had saved could not be reopened.
It also made the size advisory unreachable: the advisory and the refusal shared
the same 10,000-step threshold, and the refusal ran first, so the sentence that
exists to describe a large protocol could never be said about one.
"""

import pathlib

import pytest

from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import (
    TILING_CONFIGS,
    _build_protocol,
    _make_step,
)

# Above the advisory threshold (PROTOCOL_SIZE_ADVISORY_STEPS, 10,000) and above
# the step ceiling this suite exists to keep deleted.
LARGE_STEP_COUNT = 12_000

GLOBAL_MAX_FPS = 10.0


def _large_protocol():
    steps = [_make_step(name=f'A1_BF_{i}', label=f'custom{i:04d}') for i in range(LARGE_STEP_COUNT)]
    return _build_protocol(steps)


def _reload(protocol, tmp_path: pathlib.Path) -> Protocol:
    filepath = tmp_path / 'large_protocol.tsv'
    assert protocol.to_file(filepath) is None
    return Protocol.from_file(
        file_path=filepath,
        tiling_configs_file_loc=TILING_CONFIGS,
    )


def test_a_protocol_the_app_wrote_can_be_read_back(tmp_path):
    """The reproduced defect: the writer's own output was unreadable."""
    reloaded = _reload(_large_protocol(), tmp_path)

    assert reloaded.num_steps() == LARGE_STEP_COUNT


def test_the_size_advisory_is_reachable_on_a_loaded_protocol(tmp_path):
    """Asserted at the API tier, where the advisory's threshold and sentence live."""
    reloaded = _reload(_large_protocol(), tmp_path)

    advisory = reloaded.size_advisory(global_max_fps=GLOBAL_MAX_FPS)

    assert advisory is not None


def test_the_file_ceiling_still_refuses_and_blames_memory_not_corruption(tmp_path):
    """The resource control stays; only its diagnosis changes.

    Size is not evidence of corruption, so a legitimate large file must not be
    told it is damaged.
    """
    huge = tmp_path / 'huge_protocol.tsv'
    huge.write_bytes(b'x' * (10 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match='exceeds maximum size') as exc_info:
        Protocol.from_file(file_path=huge, tiling_configs_file_loc=TILING_CONFIGS)

    assert 'corrupt' not in str(exc_info.value).lower()


def test_a_malformed_file_is_still_refused(tmp_path):
    """Deleting the step ceiling must not make junk loadable."""
    junk = tmp_path / 'junk.tsv'
    junk.write_text('this is not a protocol\n')

    with pytest.raises(ProtocolFormatError, match='Not a valid LumaViewPro Protocol'):
        Protocol.from_file(file_path=junk, tiling_configs_file_loc=TILING_CONFIGS)
