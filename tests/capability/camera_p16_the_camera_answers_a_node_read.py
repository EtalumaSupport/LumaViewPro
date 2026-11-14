"""Camera P16: a Basler answers a read of its own named settings, idle and while it grabs.

`scope.diagnostics.read_camera_node` reads one GenICam node by the camera's
own name. It was proven on the simulator and on Basler's emulator, which
holds none of a daA3840's own nodes; this probe reads them on the body:

- `BslChunkTimestampSelector` answers `FrameStart` (the 2026-11-03 read);
- `DeviceTemperature` answers a value with its unit;
- `NumQueuedBuffers` on the `instant_camera` map answers a count;
- a name the camera has no node for answers None;
- the two tick-rate names, `TimestampTickFrequency` and
  `GevTimestampTickFrequency`, answer or are absent (a USB body is expected
  to have the first only);
- `BslEffectiveExposureTime` and `BslResultingAcquisitionFrameRate`, which
  may hold no value while the camera is idle (`CameraNodeNotReadableError`).

Each read is made idle, then again while live view grabs, which shows whether
a read during a grab is safe: the grab keeps delivering frames across the
reads. Each answer, or the refusal it met, is recorded as a figure.

    python tests/capability/camera_p16_the_camera_answers_a_node_read.py             # simulator
    python tests/capability/camera_p16_the_camera_answers_a_node_read.py --hardware  # a Basler scope

On the simulator the camera has no named settings, so every read is refused
`camera_node_unsupported` and nothing reaches the camera: the refusal is the
probe's check there. On hardware nothing is lit, nothing is saved and the
stage does not move beyond the bring-up home.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

from _lumascope.modules.exceptions import CameraNodeNotReadableError, CameraSettingUnsupportedError

# (name, node map) read idle and while grabbing.
READS = (
    ('BslChunkTimestampSelector', 'device'),
    ('DeviceTemperature', 'device'),
    ('NumQueuedBuffers', 'instant_camera'),
    ('ZZZNoSuchNode', 'device'),
    ('TimestampTickFrequency', 'device'),
    ('GevTimestampTickFrequency', 'device'),
    ('BslEffectiveExposureTime', 'device'),
    ('BslResultingAcquisitionFrameRate', 'device'),
)
# How long the grab runs before its reads, so frames are flowing.
GRAB_WARMUP_S = 2.0


def _read_all(diagnostics, when):
    """Every READS entry read once, its answer or refusal recorded under `when`."""
    answers = {}
    for name, node_map in READS:
        t0 = time.monotonic()
        try:
            got = diagnostics.read_camera_node(name, node_map)
            answer = None if got is None else {'value': got.value, 'unit': got.unit}
        except CameraNodeNotReadableError as ex:
            answer = {'not_readable_now': str(ex)}
        answers[name] = answer
        figure(
            f'{when}: {node_map}.{name}',
            {'answer': answer, 'ms': round((time.monotonic() - t0) * 1e3, 2)},
        )
    return answers


def _check_answers(answers, when):
    selector = answers['BslChunkTimestampSelector']
    check(
        f'{when}: BslChunkTimestampSelector answers FrameStart',
        selector == {'value': 'FrameStart', 'unit': ''},
        str(selector),
    )
    temperature = answers['DeviceTemperature']
    check(
        f'{when}: DeviceTemperature answers a number with its unit',
        temperature is not None
        and isinstance(temperature.get('value'), float)
        and temperature.get('unit') != '',
        str(temperature),
    )
    queued = answers['NumQueuedBuffers']
    check(
        f'{when}: NumQueuedBuffers answers a count',
        queued is not None and isinstance(queued.get('value'), int),
        str(queued),
    )
    check(
        f'{when}: a name the camera has no node for answers None',
        answers['ZZZNoSuchNode'] is None,
        str(answers['ZZZNoSuchNode']),
    )


def _frames_delivered(imaging):
    return imaging.delivered_rate()


def _probe(session):
    scope = session.scope
    figure(
        'camera',
        {
            'model': scope.capabilities.camera_model,
            'node_maps': scope.capabilities.camera_node_maps,
        },
    )
    if not HARDWARE:
        try:
            scope.diagnostics.read_camera_node('Gain')
            check('the simulated camera refuses a node read', False, 'it answered')
        except CameraSettingUnsupportedError as ex:
            check(
                'the simulated camera refuses a node read',
                ex.reason == 'camera_node_unsupported',
                ex.reason,
            )
        return

    imaging = scope.imaging
    if imaging.is_streaming():
        imaging.disable_streaming()
    check('idle: the camera is not grabbing', not imaging.is_streaming())
    _check_answers(_read_all(scope.diagnostics, 'idle'), 'idle')

    imaging.enable_streaming()
    try:
        time.sleep(GRAB_WARMUP_S)
        figure('grabbing: delivered rate before the reads', str(_frames_delivered(imaging)))
        _check_answers(_read_all(scope.diagnostics, 'grabbing'), 'grabbing')
        check('grabbing: the camera still grabs after the reads', imaging.is_streaming())
        time.sleep(GRAB_WARMUP_S)
        figure('grabbing: delivered rate after the reads', str(_frames_delivered(imaging)))
    finally:
        imaging.disable_streaming()


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _probe(session)
        return
    session, _live = make_session('probe_node_read')
    try:
        _probe(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
