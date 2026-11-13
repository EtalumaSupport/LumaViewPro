"""Camera P19: the API's frame, a saved still and the REST live JPEG are one
way up, row 0 at the top, on the connected camera.

Four modes, each a bench condition:

- target (the default): a USAF target lit in BF. The saved still and the
  live JPEG are each correlated with the API's frame as it is and turned top
  to bottom; one way up means the upright correlation is high and the turned
  one is not. The three pictures are kept beside the verdict for the eye.
- ``--dark``: the camera light-tight, 24 dB, 50 ms. The API's hot pixels
  (the mean of five frames over 30) and a saved still's (over 100) are at the
  same coordinates, not at the row-reversed ones; both lists go in the
  verdict, so two builds' runs on one body can be compared row for row.
- ``--nodes``: ``ReverseX`` and ``ReverseY`` as the camera answers them,
  taken (True) or absent (owed, turned in the driver's copy), at connect and
  again after ``set_binning(2)`` and a ``set_frame_size``, which no code path
  resets but a body's own firmware might; and the window's and the
  auto-function region's offsets and sizes at connect, so two builds' runs
  show whether a reverse moves where either is counted from. Nothing is lit,
  saved or moved.
- ``--readout``: BF off, then on while frames stream at a long exposure. A
  rolling shutter's frame lit part way through its readout is lit at the end
  read out last, so the transition frame names which end of the API frame is
  read out first; two builds' runs show whether the sensor reverse also
  reverses the readout.

    python tests/capability/camera_p19_frame_orientation.py                    # simulator
    python tests/capability/camera_p19_frame_orientation.py --hardware [--dark | --nodes | --readout] \\
        [--x-mm 62.47 --y-mm 42.47 --z 5001] [--bf-ma 3.3 --exposure-ms 0.95]

Bring-up homes the stage, so the field a person found in LumaViewPro is
given back here (plate mm and Z um, as its saved file records them).
"""

import argparse
import asyncio
import pathlib
import sys
import threading
import traceback

import cv2
import harness
import numpy as np
from harness import HARDWARE, RECORD, check, figure, hardware_session, make_session, report, void

from _lumascope.modules.coord_transformations import PlateLocation
from _lumascope.modules.exceptions import CameraSettingUnsupportedError
from _lumascope.modules.image_utils import read_tiff_with_legacy_collapse
from _lumascope.rest.live import LiveView

_parser = argparse.ArgumentParser()
_parser.add_argument('--hardware', action='store_true')
_parser.add_argument('--dark', action='store_true', help='light-tight hot-pixel mode')
_parser.add_argument('--x-mm', type=float, default=None, help='plate X of the field, mm')
_parser.add_argument('--y-mm', type=float, default=None, help='plate Y of the field, mm')
_parser.add_argument('--z', type=float, default=None, help='the field in focus, Z um')
# The 10-07 LS720 bench's conditions (Firmware docs/reference/bench/fx2/2026-10-07_ls720_flip_row8_row9.md).
_parser.add_argument('--bf-ma', type=float, default=1.0, help='target mode BF current, mA')
_parser.add_argument('--exposure-ms', type=float, default=2.0, help='target mode exposure, ms')
_parser.add_argument(
    '--nodes',
    action='store_true',
    help='the reverse nodes, at connect and after binning and frame size',
)
_parser.add_argument(
    '--readout',
    action='store_true',
    help='which end of the frame the rolling shutter reads out first',
)
ARGS = _parser.parse_args()
DARK = ARGS.dark

REVERSES = ('ReverseX', 'ReverseY')
# Where the window and the auto-function region sit on the sensor (the region
# at its selector's present entry, the one the driver configures).
PLACEMENT_NODES = (
    'OffsetX',
    'OffsetY',
    'Width',
    'Height',
    'AutoFunctionROISelector',
    'AutoFunctionROIOffsetX',
    'AutoFunctionROIOffsetY',
    'AutoFunctionROIWidth',
    'AutoFunctionROIHeight',
)
# The readout read: an exposure long against the readout, so frames span the
# LED's switch; the bands a frame is cut into across its rows; the frames
# kept after the switch.
READOUT_EXPOSURE_MS = 30.0
READOUT_BF_MA = 1.0
READOUT_BANDS = 8
READOUT_FRAMES_AFTER = 60
# A frame lit at one end and not the other: its end bands differ by at least
# this share of the lit-to-dark step.
READOUT_PART_LIT_AT_LEAST = 0.5
# The lit level stands this far above the dark one, or the frames were never
# lit and no end can be named.
READOUT_STEP_AT_LEAST_DN = 20.0
# A frame size other than bring-up's 1900 x 1900, for the read-back after a resize.
NODES_FRAME_W = NODES_FRAME_H = 1024

DARK_GAIN_DB = 24.0
DARK_EXPOSURE_MS = 50.0
DARK_FRAMES = 5
HOT_IN_MEAN = 30
HOT_IN_FILE = 100
# One way up: the pictures agree as they are and not turned (a USAF target
# turned top to bottom is a different picture).
UPRIGHT_AT_LEAST = 0.8
TURNED_AT_MOST = 0.5


def _grey(image):
    a = np.asarray(image, dtype=np.float64)
    return a.mean(axis=2) if a.ndim == 3 else a


def _correlation(a, b):
    """Pearson correlation of two pictures, b resized to a's shape."""
    a, b = _grey(a), _grey(b)
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else 0.0


def _upright_and_turned(name, frame, other):
    upright = _correlation(other, frame)
    turned = _correlation(other, frame[::-1])
    figure(
        f'{name}_correlation',
        {'upright': round(upright, 3), 'turned_top_to_bottom': round(turned, 3)},
    )
    check(
        f'{name} is the way up the API frame is',
        upright >= UPRIGHT_AT_LEAST and turned <= TURNED_AT_MOST,
        f'upright {upright:.3f}, turned {turned:.3f}',
    )


def _live_jpeg(session):
    """The REST live view's JPEG of the newest frame, through LiveView as the server serves it."""

    async def snap():
        return await LiveView(session).snapshot(None)

    got = asyncio.run(snap())
    check('the live view handed out a frame', got is not None)
    return None if got is None else got[1]


def _saved_still(session):
    stills = session.manual_capture.start_save_still().result(timeout=60)
    path = pathlib.Path(stills[0].file)
    figure('saved_file', str(path))
    return read_tiff_with_legacy_collapse(path)


def _hot(image, threshold):
    rows, cols = np.nonzero(_grey(image) > threshold)
    return {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}


def target(session):
    scope = session.scope
    session.show_channel('BF')
    scope.camera.set_exposure_ms(ARGS.exposure_ms)
    scope.illumination.led_on('BF', ARGS.bf_ma)
    try:
        frame = scope.imaging.capture_still().image
        np.save(f'{RECORD}_api_frame.npy', frame)
        cv2.imwrite(f'{RECORD}_api_frame.png', frame)
        figure('frame_shape', list(frame.shape))
        saved = _saved_still(session)
        _upright_and_turned('the saved still', frame, saved)
        jpeg = _live_jpeg(session)
        if jpeg is not None:
            pathlib.Path(f'{RECORD}_live.jpg').write_bytes(jpeg)
            live = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
            _upright_and_turned('the live JPEG', frame, live)
    finally:
        scope.illumination.leds_off()


def dark(session):
    scope = session.scope
    scope.illumination.leds_off()
    session.show_channel('BF')
    scope.camera.set_gain_db(DARK_GAIN_DB)
    scope.camera.set_exposure_ms(DARK_EXPOSURE_MS)
    frames = [
        scope.imaging.capture_still(accept_dark=True).image.astype(np.float64)
        for _ in range(DARK_FRAMES)
    ]
    mean = np.mean(frames, axis=0)
    np.save(f'{RECORD}_api_dark_mean.npy', mean)
    h = mean.shape[0]
    api_hot = _hot(mean, HOT_IN_MEAN)
    saved = _saved_still(session)
    file_hot = _hot(saved, HOT_IN_FILE)
    figure('frame_shape', list(mean.shape))
    figure('api_hot_pixels', sorted(api_hot))
    figure('saved_hot_pixels', sorted(file_hot))
    if not file_hot:
        void(
            "a saved still's hot pixels sit where the API frame's do",
            False,
            'no pixel over the threshold in the file',
        )
        return
    same = sum(p in api_hot for p in file_hot)
    turned = sum((h - 1 - r, c) in api_hot for r, c in file_hot)
    figure(
        'saved_hot_pixels_in_the_api_frame',
        {'same_place': same, 'rows_reversed': turned, 'of': len(file_hot)},
    )
    check(
        "a saved still's hot pixels sit where the API frame's do",
        same >= 0.9 * len(file_hot) and turned < same,
        f'{same} at the same place, {turned} rows reversed, of {len(file_hot)}',
    )


def _reverses(scope, when):
    """ReverseX and ReverseY as the camera answers them: True taken, None absent (owed, turned in the copy)."""
    answers = {}
    for name in REVERSES:
        got = scope.diagnostics.read_camera_node(name)
        answers[name] = None if got is None else got.value
    figure(f'{when}: reverses', answers)
    return answers


def nodes(session):
    scope = session.scope
    try:
        at_connect = _reverses(scope, 'at connect')
    except CameraSettingUnsupportedError as ex:
        void('the camera answers its reverse nodes', False, f'no named settings here: {ex.reason}')
        return
    check(
        'each reverse is taken or absent, never left off',
        all(v in (True, None) for v in at_connect.values()),
        str(at_connect),
    )
    placement = {}
    for name in PLACEMENT_NODES:
        got = scope.diagnostics.read_camera_node(name)
        placement[name] = None if got is None else got.value
    figure('at connect: the window and the auto-function region', placement)
    binning, size = scope.camera.binning, scope.camera.frame_size
    try:
        scope.camera.set_binning(2)
        after_binning = _reverses(scope, 'after set_binning(2)')
        check(
            'the reverses read the same after set_binning(2)',
            after_binning == at_connect,
            str(after_binning),
        )
        scope.camera.set_binning(binning)
        scope.camera.set_frame_size(NODES_FRAME_W, NODES_FRAME_H)
        after_size = _reverses(scope, f'after set_frame_size({NODES_FRAME_W}, {NODES_FRAME_H})')
        check(
            'the reverses read the same after set_frame_size',
            after_size == at_connect,
            str(after_size),
        )
    finally:
        scope.camera.set_binning(binning)
        scope.camera.set_frame_size(size.width, size.height)


def readout(session):
    """Which end of the API frame the rolling shutter reads out first."""
    scope = session.scope
    scope.illumination.leds_off()
    session.show_channel('BF')
    scope.camera.set_exposure_ms(READOUT_EXPOSURE_MS)
    # The dark level, settled, before the switch.
    dark = scope.imaging.capture_still(accept_dark=True).image.astype(np.float64)
    lit_from = threading.Event()
    done = threading.Event()
    profiles = []

    def listen(frame):
        if not lit_from.is_set() or done.is_set():
            return
        bands = np.array_split(_grey(frame.image).mean(axis=1), READOUT_BANDS)
        profiles.append((frame.seq, [float(b.mean()) for b in bands]))
        if len(profiles) >= READOUT_FRAMES_AFTER:
            done.set()

    scope.imaging.add_frame_listener(listen, name='camera_p19_readout')
    try:
        lit_from.set()
        scope.illumination.led_on('BF', READOUT_BF_MA)
        got = done.wait(timeout=30)
    finally:
        scope.imaging.remove_frame_listener(listen)
        scope.illumination.leds_off()
    check('the frames after the switch arrived', got, f'{len(profiles)} frames')
    if not profiles:
        return
    dark_level = float(dark.mean())
    lit_level = float(np.mean(profiles[-1][1]))
    step = lit_level - dark_level
    figure('readout_levels', {'dark': round(dark_level, 2), 'lit': round(lit_level, 2)})
    figure(
        'readout_band_means_per_frame',
        [[seq, [round(b, 1) for b in bands]] for seq, bands in profiles],
    )
    check(
        'the LED lit the frames',
        step >= READOUT_STEP_AT_LEAST_DN,
        f'lit {lit_level:.1f} against dark {dark_level:.1f}',
    )
    if step < READOUT_STEP_AT_LEAST_DN:
        return
    part_lit = [
        (seq, bands)
        for seq, bands in profiles
        if abs(bands[-1] - bands[0]) >= READOUT_PART_LIT_AT_LEAST * step
    ]
    check(
        'a frame lit part way through its readout was caught',
        bool(part_lit),
        f'{len(part_lit)} of {len(profiles)} frames, step {step:.1f}',
    )
    if not part_lit:
        return
    ends = {
        'last row (row 0 read out first)' if b[-1] > b[0] else 'row 0 (the last row read out first)'
        for _, b in part_lit
    }
    figure('readout_lit_end', sorted(ends))
    check('every part-lit frame is lit at the same end', len(ends) == 1, str(sorted(ends)))


def body(session):
    if ARGS.nodes:
        nodes(session)
        return
    if ARGS.x_mm is not None or ARGS.y_mm is not None:
        session.scope.motion.move_to_plate(PlateLocation(ARGS.x_mm, ARGS.y_mm, None))
    if ARGS.z is not None:
        session.scope.motion.move_z(ARGS.z)
    (readout if ARGS.readout else dark if DARK else target)(session)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            body(session)
        sys.exit(report())
    session, _live = make_session('camera_p19', home=True, microscope='LS720')
    try:
        body(session)
    except BaseException:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        session.shutdown()
    sys.exit(report())


if __name__ == '__main__':
    harness.banner(
        f'camera_p19 frame orientation ({"nodes" if ARGS.nodes else "readout" if ARGS.readout else "dark" if DARK else "target"})'
    )
    main()
