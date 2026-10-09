# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run captures in the image mode the settings hold.

The image mode has one store, ``settings['image_mode']``, and one writer,
``session.set_image_mode``. A run reads it from the settings it starts
with, as the GUI's Run always did, so a script that chose 12-bit through
the Session gets 12-bit files without restating the mode to the run.
"""

from modules.image_utils import load_pixels
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60.0


def test_a_run_saves_the_mode_the_session_was_set_to(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.set_image_mode('12bit_scientific')

        run = runner.run_single_scan(
            protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
        )
        outcome = run.wait(timeout_s=WAIT_S)
        assert outcome is not None and outcome.status == 'completed', outcome
        assert run.wait_for_files(timeout_s=WAIT_S) is not None, "the run's files never landed"

    saved = sorted((tmp_path / 'runs').rglob('*.tif*'))
    assert saved, 'the run saved no image'
    pixels, significant_bits = load_pixels(saved[0])
    assert significant_bits == 12, f'{saved[0].name} holds {significant_bits}-bit pixels'
    assert pixels.dtype.itemsize == 2
