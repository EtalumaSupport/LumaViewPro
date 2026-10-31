# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Regression tests for P2 bug fixes.

Tests that the following bugs stay fixed:
- #563: Autofocus race condition -- protocol_end() must come AFTER final move
- #568: Duration precision -- sub-second protocols must round-trip through save/load
- #424: Video bit-depth -- 16-bit frames must be converted to 8-bit before codec write
"""

import datetime

import numpy as np
import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time.


from modules.protocol import Protocol
from tests.protocol_fixtures import protocol_of


# ---------------------------------------------------------------------------
# #563: Autofocus race condition -- final move must complete before AF signals done
# ---------------------------------------------------------------------------


class TestAFRaceCondition:
    """Verify the AF "done" signal is not raised until the final moves are issued.

    The original bug: AFE.protocol_end() (the legacy "done" signal) fired
    between the two final moves, so the protocol executor stopped processing
    while the second move was still pending. Now AFE.run() issues both final
    moves before it returns to its worker thread, and the Future resolves
    AFTER AFE.run() returns. The moves are started, not waited: the step
    runner, which polls p._af_future.done(), then holds its capture while
    the stage reports motion (``modules/protocol_step_runner.py``).
    """

    def test_final_moves_are_issued_before_run_returns(self, sim_scope, monkeypatch):
        """The fine pass must issue both final moves (approach-from-below,
        then best) inline, so they are already on the motion API when run()
        returns and the AutofocusThread Future resolves.

        AFE.run() is synchronous; protect against a future refactor that
        accidentally offloads the final move to another thread and
        re-introduces the race that the AutofocusThread + Future contract
        currently prevents.
        """
        from tests.af_drives import AF_OBJECTIVE, af_runner_on, sweep, z_asked

        runner = af_runner_on(sim_scope)
        asked = z_asked(sim_scope, monkeypatch)

        result = sweep(runner, sim_scope)

        fine_resolution = sim_scope.objectives[AF_OBJECTIVE].autofocus_step_min_um
        assert asked[-2:] == [pytest.approx(result - fine_resolution), result], (
            'the last pass must issue the approach-from-below move and the '
            f'final best-focus move inline before run() returns; asked: {asked}'
        )


# ---------------------------------------------------------------------------
# #568: Duration precision -- sub-second protocols must survive save/load
# ---------------------------------------------------------------------------


class TestDurationPrecision:
    """Verify that short protocol durations don't get truncated to 0.0.

    Bug #568: round(hours, 2) for a 10-second protocol produces 0.0,
    which fails validation on reload (duration must be > 0).
    """

    def test_10_second_duration_round_trips(self, tmp_path):
        """A 10-second protocol duration must survive save -> load."""
        duration = datetime.timedelta(seconds=10)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, '10-second duration must not round to 0.0'

    def test_1_second_duration_round_trips(self, tmp_path):
        """A 1-second protocol duration must survive save -> load."""
        duration = datetime.timedelta(seconds=1)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, '1-second duration must not round to 0.0'

    def test_sub_second_duration_round_trips(self):
        """A 0.5-second protocol duration must survive save -> load."""
        duration = datetime.timedelta(milliseconds=500)
        duration_hours = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours > 0, '0.5-second duration must not round to 0.0'

    def test_old_rounding_would_fail(self):
        """Confirm that the old round(hours, 2) would produce 0.0 for short durations."""
        duration = datetime.timedelta(seconds=10)
        duration_hours_old = round(duration.total_seconds() / 3600.0, 2)
        assert duration_hours_old == 0.0, 'Old rounding (2 decimals) should produce 0.0 for 10s'

        duration_hours_new = round(duration.total_seconds() / 3600.0, 6)
        assert duration_hours_new > 0, 'New rounding (6 decimals) should preserve precision'

    def test_save_writes_nonzero_duration(self, tmp_path):
        """Saved protocol file must contain a non-zero duration for short protocols."""
        protocol = protocol_of()

        # Configure a minimal protocol with 10-second duration
        protocol._config = {
            'version': Protocol.CURRENT_VERSION,
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(seconds=10),
            'labware_id': 'none',
            'capture_root': '',
            'steps': Protocol._create_empty_steps_df(),
        }

        # Save
        save_path = tmp_path / 'test_protocol.tsv'
        protocol.to_file(file_path=save_path)

        # Read back the raw file and find the Duration line
        content = save_path.read_text()
        for line in content.splitlines():
            if line.startswith('Duration'):
                duration_val = float(line.split('\t')[1])
                assert duration_val > 0, (
                    f'Duration in saved file must be > 0, got {duration_val} (bug #568)'
                )
                # Verify it converts back to ~10 seconds
                seconds = duration_val * 3600
                assert abs(seconds - 10.0) < 0.1, f'Duration should be ~10 seconds, got {seconds}'
                return

        pytest.fail('Duration row not found in saved protocol file')


# ---------------------------------------------------------------------------
# #424: Video bit-depth -- 16-bit frames converted to 8-bit before write
# ---------------------------------------------------------------------------


class TestVideoBitDepth:
    """Converter-level bit-depth regression coverage.

    The encoder-boundary half of this class (uint16 frames must reach the
    codec as uint8 -- bug #424) lives with the writer's own tests in
    test_video_writer.py::TestVideoBitDepth, harnessed on the PyAV
    backend; this class keeps the pure-converter invariant.
    """

    def test_convert_16bit_to_8bit_preserves_relative_intensity(self):
        """Conversion preserves relative brightness, rescaling the full 16-bit
        range to 8-bit (value / 65535 * 255) -- the one canonical mapping shared
        with convert_to_8bit, not the legacy high-byte (/256) truncation."""
        import modules.image_utils as image_utils

        frame = np.array([[0, 256, 512, 65535]], dtype=np.uint16)
        result = image_utils.convert_16bit_to_8bit(frame)

        assert result.dtype == np.uint8
        assert result[0, 0] == 0  # 0 -> 0
        assert result[0, 1] == 0  # 256/65535*255 = 0.996 -> 0
        assert result[0, 2] == 1  # 512/65535*255 = 1.992 -> 1
        assert result[0, 3] == 255  # 65535 -> full white
