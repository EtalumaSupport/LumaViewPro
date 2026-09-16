"""A dark capture must say it was dark, not that the camera is gone.

Bench, 2026-09-15: a bright-field capture at 0.01 ms was correctly rejected by
the dark-floor guard -- 14 us of integration IS black -- and the user was told

    'camera inactive or not grabbing. Check the live view, then try again.'

while the camera took gain and exposure writes in that same millisecond and the
run held no disconnect of any kind. The operator went looking for a reconnect
fault that did not exist.

The guard is right and stays: issues #671 and #721 are the record of a near-black
frame being silently saved, and `tests/test_dark_floor_capture_guard.py` locks the
rejection itself. What was wrong is the CAUSE reported afterwards.

`capture_failure_cause` is a ladder of named causes whose own docstring warns
that 'a hardcoded cause once mislabeled every failure as the latter'. The dark
path had no rung, so it fell through to the camera-inactive default -- exactly
the failure the docstring predicted. This adds the missing rung and locks it.
"""

import sys
from collections import deque
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('modules.settings_init', MagicMock())


def test_a_dark_rejection_names_darkness_not_a_dead_camera():
    """The rung that was missing."""
    from modules.lumascope_api.imaging import capture_failure_cause

    cause = capture_failure_cause({'dark_rejected': True})

    assert cause != 'camera inactive or not grabbing', (
        'a dark frame reported as a dead camera is what sent an operator '
        'hunting a reconnect fault that did not exist'
    )
    assert 'dark' in cause.lower(), f'the cause must name darkness; got {cause!r}'


def test_every_rung_still_outranks_the_default():
    """The ladder's other causes keep their own answers."""
    from modules.lumascope_api.imaging import capture_failure_cause

    default = capture_failure_cause({})
    assert default == 'camera inactive or not grabbing'
    for info in (
        {'deadline_expired': True},
        {'drain_failed': True},
        {'chunk_rejected': 'exposure'},
        {'dark_rejected': True},
    ):
        assert capture_failure_cause(info) != default, (
            f'{info} must name its own cause rather than fall through'
        )


def test_the_dark_flag_is_cleared_per_capture():
    """A stale flag would misattribute the NEXT failure.

    The guard sets the flag deep in the grab loop and the caller reads it after;
    without a reset at the top of each capture, one dark capture would make
    every later failure claim darkness.
    """
    import inspect

    from modules.lumascope_api.imaging import ImagingAPI

    src = inspect.getsource(ImagingAPI._capture_and_wait_impl)
    assert 'self._dark_rejected = False' in src, (
        'capture_and_wait must clear the dark flag before each capture'
    )


def test_the_guard_itself_is_untouched():
    """The rejection stays -- #671 / #721 are the record of why.

    This fix names the cause; it does not relax the guard. A future change that
    turns the rejection into a silent save re-opens both issues.
    """
    import inspect

    from modules.lumascope_api.imaging import ImagingAPI

    src = inspect.getsource(ImagingAPI._get_image_impl)
    block = src.split('if dark_floor_check:', 1)[1].split('if verify_chunk_targets:', 1)[0]
    assert 'return None' in block, (
        'the dark-floor guard must still reject -- a near-black frame that is '
        'silently saved is issues #671 and #721'
    )


@pytest.mark.parametrize(
    'interval_ms,exposure_ms,median_ms,should_warn',
    [
        (1000.0, 1000.0, 70.0, False),  # the bench case: 1 fps at a 1 s exposure
        (982.0, 1000.0, 70.0, False),  # just under one exposure
        (2500.0, 1000.0, 70.0, True),  # stalled well past the exposure
        (500.0, 2.0, 70.0, True),  # short exposure, a real spike
    ],
)
def test_slow_frame_waits_one_exposure(interval_ms, exposure_ms, median_ms, should_warn):
    """A frame cannot arrive sooner than it takes to expose.

    At 1000 ms the camera delivers ~1 fps while the rolling median still
    reflects the previous short exposure, so every interval read as a spike --
    25 warnings in one bench run for the camera doing what it was told. The
    exposure is a FLOOR on the threshold, so a genuine stall still reports.
    """
    import inspect

    from ui.scope_display import FRAME_SPIKE_FLOOR_MS, FRAME_SPIKE_RATIO, ScopeDisplay

    check_src = inspect.getsource(ScopeDisplay._check_slow_frame)
    assert '_exposure_floor_ms()' in check_src, (
        'the threshold must consult the exposure; a formula retyped here would '
        'pass with or without the fix'
    )
    threshold_ms = max(FRAME_SPIKE_FLOOR_MS, FRAME_SPIKE_RATIO * median_ms, exposure_ms)
    assert (interval_ms > threshold_ms) is should_warn


def test_exposure_floor_reads_the_api_and_degrades_quietly():
    """The exposure the CAMERA runs bounds delivery; the slider can differ."""
    import inspect

    from ui.scope_display import ScopeDisplay

    src = inspect.getsource(ScopeDisplay._exposure_floor_ms)
    assert 'exposure_ms_cached' in src, (
        'the floor comes from the API cache -- the widget displays that value, '
        'it is not the authority on what the camera is doing'
    )
    assert 'return 0.0' in src, (
        'an unreadable exposure must leave the median-based threshold unchanged'
    )


def test_a_slow_frame_at_a_long_exposure_is_not_logged():
    """End to end through the real method, not the formula."""
    from ui.scope_display import FRAME_SPIKE_WINDOW, ScopeDisplay

    class _Stand:
        _check_slow_frame = ScopeDisplay._check_slow_frame
        _spike_median = ScopeDisplay._spike_median

        def __init__(self, exposure_ms):
            self._exposure = exposure_ms
            self._spike_interval_window = deque(
                [70.0] * FRAME_SPIKE_WINDOW, maxlen=FRAME_SPIKE_WINDOW
            )
            self._last_ok_frame_time = 1000.0
            self._last_ok_compute = (1.0, 1.0, 1.0)
            self._spike_median_cache = None
            self._spike_median_refresh = 0.0
            self._slow_frame_last_log = 0.0

        def _exposure_floor_ms(self):
            return self._exposure

    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger('LVP')
    handler = _Capture()
    logger.addHandler(handler)
    try:
        # One second between frames at a one second exposure: correct, not slow.
        stand = _Stand(exposure_ms=1000.0)
        stand._check_slow_frame(1001.0, grab_ms=1.0, proc_ms=1.0, eng_ms=1.0)
        assert not [r for r in records if 'SLOW FRAME' in r], (
            'a frame interval explained by the exposure must not warn'
        )
    finally:
        logger.removeHandler(handler)
