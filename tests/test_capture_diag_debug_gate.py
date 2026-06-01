# Copyright Etaluma, Inc.
"""Regression test: the per-step CAPTURE DIAG camera reads are debug-gated.

The #610 capture diagnostic compares the step's intended gain/exposure
against the camera's ACTUAL (live) values, so the reads must stay live --
but they were firing on every protocol step even in normal operation,
because the f-string arguments evaluate before logger.debug decides to drop
the line. The two live SDK reads are now gated on debug being enabled.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_capture_diag_camera_reads_gated_on_debug():
    src = (REPO_ROOT / 'modules' / 'protocol_image_writer.py').read_text()
    idx = src.find('[CAPTURE DIAG]')
    assert idx != -1, 'CAPTURE DIAG diagnostic not found'
    window = src[max(0, idx - 500) : idx]
    assert 'isEnabledFor(logging.DEBUG)' in window, (
        'the live get_gain()/get_exposure_time() reads feeding the CAPTURE '
        'DIAG line must be gated on logger.isEnabledFor(logging.DEBUG) so '
        'they do not run every step when the debug line is dropped'
    )
    assert 'get_gain()' in window and 'get_exposure_time()' in window, (
        'the diagnostic must still read the live camera values when enabled '
        '(comparing intended vs actual is the whole point)'
    )
