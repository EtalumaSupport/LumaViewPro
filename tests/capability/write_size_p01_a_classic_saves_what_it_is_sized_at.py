"""The protocol write size's V4 (PROTOCOL_WRITE_SIZE_PLAN_2026-11-16, section 7):
on a Classic (FX2) scope, every image mode saves the bytes a frame is sized at.

The sizing rule (``write_size.FrameSizing.saved_frame_bytes``) is pinned
against the writers on the simulated FX2 (``tests/test_write_size.py``); the
plan's first pass found the simulator's FX2 may not match the hardware. Here
the same cases run on the real camera: in each image mode, a single frame and
a sum of 3, a channel shown in colour with false colour on and off and a grey
one, the rule's bytes against the array the run's still writer hands the TIFF
(the capture at the mode's depth, then the save encoding's false colour); and
in each mode one still saved through the manual-capture path, the rule's
bytes against the saved file's pixels.

    python tests/capability/write_size_p01_a_classic_saves_what_it_is_sized_at.py
    python tests/capability/write_size_p01_a_classic_saves_what_it_is_sized_at.py --hardware
"""

import sys
import traceback

import harness
from _lumascope.modules.scope_parts import Board
import numpy as np
import tifffile
from harness import HARDWARE, check, figure, hardware_session, report

from _lumascope.modules import channel_record, image_mode, image_utils
from _lumascope.modules.scope_capabilities import FrameSize
from _lumascope.modules.write_size import FrameSizing

CASES = (('Ex488', True), ('Ex488', False), ('BF', True))
SUMS = (1, 3)
BF_PERCENT = 5


def _sizing(session, mode, frame):
    return FrameSizing.for_camera(
        frame,
        session.scope.capabilities,
        image_mode.ImageCaptureConfig.from_image_mode(mode),
        session.channel_colors,
    )


def _the_writers_array(session, mode, sum_count, channel, false_color_on):
    """The pixels a run's still writer hands the TIFF, as it builds them."""
    config = image_mode.ImageCaptureConfig.from_image_mode(mode)
    still = session.scope.imaging.capture_still(
        force_to_8bit=config.capture_depth == 8,
        sum_count=sum_count,
        accept_dark=True,
        timeout_s=10.0,
    )
    color = channel_record.shown_color(channel, false_color_on, session.channel_colors)
    return still.image, image_utils.maybe_apply_false_color(
        data=still.image,
        color=color,
        use_false_color_16bit=config.save_encoding == image_mode.SAVE_ENCODING_RGB,
    )


def body(session):
    scope = session.scope
    figure(
        'the camera',
        {
            'camera_max_significant_bits': scope.capabilities.camera_max_significant_bits,
            'is_color_native': scope.capabilities.is_color_native,
            'frame': scope.camera.frame_size.to_dict(),
        },
    )
    check(
        'the camera states its depth (a run is sized only when it does)',
        scope.capabilities.camera_max_significant_bits is not None,
    )
    session.show_channel('BF')
    scope.illumination.led_on('BF', BF_PERCENT)
    offenders = []
    try:
        for mode in image_mode.ImageMode:
            scope.camera.set_image_mode(mode)
            figure(f'{mode} pixel_format', scope.camera.pixel_format)
            for sum_count in SUMS:
                for channel, false_color_on in CASES:
                    raw, saved = _the_writers_array(
                        session, mode, sum_count, channel, false_color_on
                    )
                    sized = _sizing(
                        session, mode, FrameSize(saved.shape[1], saved.shape[0])
                    ).saved_frame_bytes(
                        channel=channel, sum_count=sum_count, false_color_on=false_color_on
                    )
                    figure(
                        f'{mode} sum {sum_count} {channel} false colour {false_color_on}',
                        {
                            'captured': f'{raw.dtype} {raw.shape} max {int(np.max(raw))}',
                            'saved': f'{saved.dtype} {saved.shape}',
                            'saved_bytes': saved.nbytes,
                            'sized_bytes': sized,
                        },
                    )
                    if sized != saved.nbytes:
                        offenders.append(
                            f'{mode} sum {sum_count} {channel} false colour {false_color_on}: '
                            f'sized {sized}, the writer saves {saved.nbytes}'
                        )

            stills = session.manual_capture.start_save_still().result(timeout=60)
            saved_file = stills[0].file
            if saved_file.suffix.lower() not in ('.tif', '.tiff'):
                harness.void(
                    f'{mode} saved still', False, f'saved as {saved_file.suffix}, not a TIFF'
                )
                continue
            bf = session.settings['BF']
            with tifffile.TiffFile(saved_file) as tif:
                page = tif.pages[0]
                file_bytes = int(np.prod(page.shape)) * page.dtype.itemsize
            sized = _sizing(session, mode, scope.camera.frame_size).saved_frame_bytes(
                channel='BF', sum_count=bf['sum'], false_color_on=bf['false_color']
            )
            figure(
                f'{mode} saved still',
                {
                    'file': saved_file.name,
                    'page': f'{page.dtype} {page.shape}',
                    'sum': bf['sum'],
                    'false_color': bf['false_color'],
                    'file_pixel_bytes': file_bytes,
                    'sized_bytes': sized,
                },
            )
            if sized != file_bytes:
                offenders.append(
                    f'{mode} saved still (BF, sum {bf["sum"]}): sized {sized}, '
                    f'the file holds {file_bytes}'
                )
    finally:
        scope.illumination.led_off('BF')
    check(
        'every case saves the bytes the rule sizes it at',
        not offenders,
        '\n'.join(offenders),
    )


def main():
    if HARDWARE:
        with hardware_session(needs=(Board.CAMERA,)) as (session, _runner):
            body(session)
        sys.exit(report())
    session, _live = harness.make_session('write_size_p01', home=True, microscope='LS620')
    try:
        body(session)
    except BaseException:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        session.shutdown()
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
