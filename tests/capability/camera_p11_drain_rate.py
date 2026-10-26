"""Camera P11: the IDS frame rate with LumaViewPro's unpack out of the loop.

At 2x2 binning the U3-34L delivers 46 to 54 fps under LumaViewPro and 66 in
IDS Cockpit, with the camera set to 66 in both and no frame counted lost
(camera P10, 2026-10-20): the frames are never started on the host. This
splits the cause. Each size is read twice: the normal pipeline through the
diagnostic probe (the stream's own delivered count over 10 s), then a drain
in which the unpack worker is paused and a loop only waits for each finished
buffer and hands it straight back to the camera (the driver's
`_drain_finished_buffers`, the unpack benchmark's own loop), timed over
DRAIN_FRAMES. A drain at 66 puts the gap in LumaViewPro's pipeline; a drain
at the normal rate puts it in the stream's configuration.

The first run (2026-10-20, 3.95 camera) drained at 55 against a normal 53,
so the gap is in the stream's configuration, and the probe now measures that
itself. Every readable node of the camera's three node maps (the remote
device, the data stream, the device's transport layer) is recorded at 2x2
before and after its drain, and the nodes whose value changed across the
drain are named, which shows any frame counter the camera keeps. Then the
sweep: at 2x2, for each pair of the data stream's transfer request count and
bulk transfer size from the current values up to what the nodes allow (a
smaller one only slows the stream), the stream is restarted with
the pair written, the values read back, and a drain timed. A pair that
drains at the camera's maximum names the setting to change; none doing so
leaves the buffer count (fixed in the driver at the minimum plus 5) and the
camera's own settings, which the dump holds for comparison with Cockpit.

The second sweep run (2026-10-20) found no transfer pair faster (46 to 50
fps), with all 8 buffers queued and empty: the host waits on the camera.
Cockpit's 66 was at a 2 ms exposure, the probe's runs at 5 ms, so the
exposure sweep follows: at 2x2, a drain at each of SWEEP_EXPOSURES_MS
beside the frame rate the camera declares there.

The exposure changed nothing (47 to 50 fps at every exposure), and each
buffer's camera frame ID and timestamp then showed where the frames go: no
frame is lost on the way to LumaViewPro; at 2x2 the camera itself skips whole
sensor slots, its frames 15 or 30 ms apart. The slot hunt follows: at 2x2, a
timed drain that keeps each buffer's frame ID and camera timestamp (the
camera's own rate, the slots it skipped, any frame-ID gap, how far behind the
camera the host receives) at each AcquisitionFrameRate step below the maximum,
at each throughput limit with the camera stopped and unlocked around the write,
at the full-depth pixel format and over the whole sensor. Its first run
(2026-10-21) put the cause in the frame rate: at the exact maximum, which at
2x2 is the throughput limit's, the camera skips 40 to 50% of its slots; a
few percent below it, none. One run left the margin uncertain (0.98 of the
maximum was clean once and skipped a fifth of its slots once), so the margin
run follows: at 2x2 and 1x1, in both pixel formats, MARGIN_REPEATS drains at
each fraction of the maximum, and the highest fraction clean in every repeat.
Then the other way to set the rate, IDS's AcquisitionFrameRateTarget, at
fractions of the maximum with the same repeats.

    python tests/capability/camera_p11_drain_rate.py --hardware  # the connected IDS scope

The drain, the dump and the sweep reach the driver directly, as the unpack
benchmark does: they are a bench instrument, not API members. While a drain
runs the live view stores no frame, so the stream check may report one
stall; that report is expected. The stream's transfer settings are restored
at the end. The simulated camera has no drain, so the simulator run is a
void. The checks, the figures and the dumps are in the run's verdict file
(the harness's THE RECORD).
"""

import itertools
import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void
from drivers import host_clock

EXPOSURE_MS = 5
WINDOW_S = 10
SETTLE_S = 3
DRAIN_FRAMES = 300


def _normal(scope, label):
    time.sleep(SETTLE_S)
    probe = scope.diagnostics.run_pylon_diagnostic_probe(WINDOW_S)
    derived = probe.get('derived', {})
    deltas = probe.get('deltas', {})
    row = {
        'observed_fps': derived.get('observed_fps'),
        'is_camera_rate': derived.get('observed_fps_is_camera_rate'),
        'started': deltas.get('StreamStartedFrameCount'),
        'delivered': deltas.get('StreamDeliveredFrameCount'),
        'fps_max': probe.get('config', {}).get('acquisition_frame_rate_max'),
        'frame': scope.camera.read_frame_size().to_dict(),
        'binning': scope.camera.read_binning(),
    }
    figure(f'{label}_normal', row)
    return row


def _drain(scope, label):
    driver = scope._camera_driver
    if not hasattr(driver, '_drain_finished_buffers'):
        void(f'{label}: the drain runs', False, 'this camera driver has no drain (the simulator)')
        return None
    frames = []
    started = time.monotonic()
    driver._drain_finished_buffers(
        DRAIN_FRAMES, lambda img: frames.append(time.monotonic()), label='drain-rate'
    )
    span = (frames[-1] - frames[0]) if len(frames) > 1 else 0.0
    row = {
        'frames': len(frames),
        'of': DRAIN_FRAMES,
        'fps': (len(frames) - 1) / span if span > 0 else None,
        'wall_s': round(time.monotonic() - started, 2),
    }
    figure(f'{label}_drain', row)
    check(f'{label}: the drain received frames', len(frames) > 1, str(row))
    return row


def _nodes(driver, nodemap):
    """Every node of one node map: presence, access, value or entries, and range."""
    nodes = {}
    for node in nodemap.Nodes():
        name = node.Name()
        info = driver._diag_probe_node(nodemap, name)
        for bound in ('Minimum', 'Maximum', 'Increment'):
            try:
                info[bound.lower()] = getattr(nodemap.FindNode(name), bound)()
            except Exception:
                pass
        nodes[name] = info
    return nodes


def _dump(driver):
    maps = {
        'remote': lambda: driver.remote_nodemap,
        'stream': lambda: driver.data_stream.NodeMaps()[0],
        'device': lambda: driver.active.NodeMaps()[0],
    }
    dump = {}
    for key, nodemap in maps.items():
        try:
            dump[key] = _nodes(driver, nodemap())
        except Exception as e:
            dump[key] = {'_access_error': f'{type(e).__name__}: {e}'}
    return dump


def _changed(before, after):
    """The nodes whose value differs between two dumps, as (before, after)."""
    changed = {}
    for key, nodes in after.items():
        for name, info in nodes.items():
            was = before.get(key, {}).get(name)
            if (
                isinstance(info, dict)
                and isinstance(was, dict)
                and 'value' in info
                and info['value'] != was.get('value')
            ):
                changed[f'{key}.{name}'] = (was.get('value'), info['value'])
    return changed


_TRANSFER = ('U3vStreamChannelTransferRequestCount', 'U3vStreamChannelBulkTransferSize')


def _steps(info):
    """The values a sweep tries for one node: from its current value up to its maximum.

    Only upward: the current setting already falls short, and a smaller one
    only slows the stream (the minimums, one 1024-byte transfer, delivered
    nothing in the first sweep, each drain running out its 300 waits). The
    value doubles, kept on the node's increment.
    """
    current, top = int(info['value']), int(info['maximum'])
    step = int(info.get('increment') or 1)
    values, size = {current, top}, current * 2
    while size < top:
        values.add(size - size % step)
        size *= 2
    return sorted(values)


def _sweep(scope, dump):
    driver = scope._camera_driver
    stream = dump.get('stream', {})
    count_info = stream.get(_TRANSFER[0], {})
    size_info = stream.get(_TRANSFER[1], {})
    figure('sweep_nodes', {'count': count_info, 'size': size_info})
    if not count_info.get('present') or not size_info.get('present'):
        check('the transfer nodes are present to sweep', False, str(stream.get('_access_error')))
        return
    originals = (int(count_info['value']), int(size_info['value']))
    rows = []
    try:
        for count in _steps(count_info):
            for size in _steps(size_info):
                with driver.update_camera_config():
                    wrote = (
                        driver.set_num_max_queued_urbs(count),
                        driver.set_max_transfer_size(size),
                    )
                nodemap = driver.data_stream.NodeMaps()[0]
                read = tuple(driver._diag_probe_node(nodemap, n).get('value') for n in _TRANSFER)
                print(f'SWEEP: count {count}, size {size}, read back {read}', flush=True)
                time.sleep(SETTLE_S)
                drain = _drain(scope, f'sweep_c{count}_s{size}')
                rows.append(
                    {
                        'count': count,
                        'size': size,
                        'wrote': wrote,
                        'read_back': read,
                        'drain': drain,
                    }
                )
    finally:
        with driver.update_camera_config():
            driver.set_num_max_queued_urbs(originals[0])
            driver.set_max_transfer_size(originals[1])
    figure('sweep', rows)
    best = max(
        (r for r in rows if r['drain'] and r['drain']['fps']),
        key=lambda r: r['drain']['fps'],
        default=None,
    )
    figure('sweep_best', best)


# Cockpit showed 66 fps at 2x2 with a 2 ms exposure; the probe's 5 ms got 48.
SWEEP_EXPOSURES_MS = (0.5, 1, 2, 3, 5, 8, 12)


def _exposure_sweep(scope):
    """At 2x2, the drain rate at each exposure beside the rate the camera declares.

    A rate that follows 1 / (exposure + the sensor's frame time) is a camera
    that does not overlap exposure with readout here; a rate below the
    declared maximum at every exposure is not set by the exposure at all.
    """
    rows = []
    for exposure_ms in SWEEP_EXPOSURES_MS:
        written = scope.camera.set_exposure_ms(exposure_ms)
        config = scope.diagnostics.run_pylon_diagnostic_probe(0).get('config', {})
        print(f'EXPOSURE SWEEP: {written} ms', flush=True)
        time.sleep(SETTLE_S)
        rows.append(
            {
                'exposure_ms': written,
                'declared_fps': config.get('acquisition_frame_rate'),
                'declared_fps_max': config.get('acquisition_frame_rate_max'),
                'drain': _drain(scope, f'exposure_{exposure_ms}ms'),
            }
        )
    figure('exposure_sweep', rows)
    scope.camera.set_exposure_ms(EXPOSURE_MS)


# The slot hunt and the margin run. A skipped slot is a frame interval of two or
# more sensor frame times (SensorFrameTimeClocks over the sensor clock) with no
# frame-ID gap: the camera made no frame there, so no host setting recovers it.

_KEY_NODES = (
    'AcquisitionFrameRate',
    'DeviceLinkAcquisitionFrameRateLimit',
    'DeviceLinkThroughputLimit',
    'DeviceLinkCurrentThroughput',
    'DeviceLinkThroughputLimitComponentValueControl',
    'SensorFrameTimeClocksValueControl',
    'SensorFrameTimeClocksMinControl',
    'SensorClockFrequencyValueControl',
    'SensorLineTimeClockValueControl',
    'ExposureTime',
    'Width',
    'Height',
    'OffsetX',
    'OffsetY',
    'BinningHorizontal',
    'BinningVertical',
    'PixelFormatValueControl',
    'PayloadSize',
    'TLParamsLocked',
    'AcquisitionFrameRateTargetEnable',
)
AFR_STEPS = ('max', 0.9999, 0.999, 0.995, 0.99, 0.98, 0.96, 0.93, 0.9, 0.8)
DLTL_STEPS_BPS = (
    None,
    290_000_000,
    280_000_000,
    270_000_000,
    250_000_000,
    234_000_000,
    200_000_000,
    150_000_000,
)


def _node(driver, name):
    return driver.remote_nodemap.FindNode(name)


def _registers(driver):
    regs = {}
    for name in _KEY_NODES:
        try:
            regs[name] = _node(driver, name).Value()
        except Exception as e:
            regs[name] = f'{type(e).__name__}: {e}'
    for name in ('AcquisitionFrameRate', 'DeviceLinkThroughputLimit', 'Width', 'Height'):
        try:
            regs[f'{name}.max'] = _node(driver, name).Maximum()
        except Exception as e:
            regs[f'{name}.max'] = f'{type(e).__name__}: {e}'
    try:
        regs['PixelFormat'] = _node(driver, 'PixelFormat').CurrentEntry().SymbolicValue()
    except Exception as e:
        regs['PixelFormat'] = f'{type(e).__name__}: {e}'
    return regs


def _member(buffer, name):
    try:
        return getattr(buffer, name)()
    except Exception as e:
        return repr(e)


def _timed_drain(driver, n_frames):
    """The drain with each buffer's camera frame ID and timestamp kept."""
    handler = driver.cam_image_handler
    if handler is not None:
        handler.stop()
    try:
        driver.data_stream.FlushPendingKillWaits()
    except Exception:
        pass
    frames, waits_failed = [], 0
    try:
        for _ in range(n_frames):
            try:
                buffer = driver.data_stream.WaitForFinishedBuffer(2000)
            except Exception:
                waits_failed += 1
                continue
            try:
                frames.append(
                    (
                        _member(buffer, 'FrameID'),
                        _member(buffer, 'Timestamp_ns'),
                        host_clock.steady_ns(),
                        _member(buffer, 'IsIncomplete'),
                    )
                )
            finally:
                driver.data_stream.QueueBuffer(buffer)
    finally:
        if handler is not None:
            handler.start()
    return frames, waits_failed


def _stats(frames, waits_failed, slot_ms):
    ints = [f for f in frames if isinstance(f[0], int) and isinstance(f[1], int)]
    row = {'frames': len(frames), 'waits_failed': waits_failed}
    row['incomplete'] = sum(1 for f in frames if f[3] is True)
    if len(ints) < 2:
        row['unreadable'] = frames[:3]
        return row
    ids = [f[0] for f in ints]
    ts = [f[1] for f in ints]
    host = [f[2] for f in ints]
    gaps = [b - a for a, b in itertools.pairwise(ids)]
    dts = [(b - a) / 1e6 for a, b in itertools.pairwise(ts)]
    row['camera_fps'] = round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9), 2)
    row['host_fps'] = round((len(host) - 1) / ((host[-1] - host[0]) / 1e9), 2)
    row['frame_id_gaps'] = sum(1 for g in gaps if g != 1)
    row['frame_id_gap_max'] = max(gaps)
    row['dt_ms_min_median_max'] = (
        round(min(dts), 2),
        round(sorted(dts)[len(dts) // 2], 2),
        round(max(dts), 2),
    )
    if slot_ms:
        slots = {}
        for d in dts:
            k = round(d / slot_ms)
            slots[k] = slots.get(k, 0) + 1
        row['slots'] = dict(sorted(slots.items()))
        row['slot_ms'] = round(slot_ms, 3)
    lag = [(h - t) / 1e6 for t, h in zip(ts, host, strict=True)]
    low = min(lag)
    row['host_lag_ms_median_max'] = (
        round(sorted(x - low for x in lag)[len(lag) // 2], 2),
        round(max(x - low for x in lag), 2),
    )
    return row


def _point(scope, label, settle=True):
    driver = scope._camera_driver
    if settle:
        # A write's frames are past before the drain starts; a repeat follows at once.
        time.sleep(SETTLE_S)
    regs = _registers(driver)
    try:
        slot_ms = (
            1000.0
            * regs['SensorFrameTimeClocksValueControl']
            / regs['SensorClockFrequencyValueControl']
        )
    except Exception:
        slot_ms = None
    try:
        temps = driver.get_all_temperatures()
    except Exception as e:
        temps = f'{type(e).__name__}: {e}'
    frames, waits_failed = _timed_drain(driver, DRAIN_FRAMES)
    row = {'registers': regs, 'temperatures': temps, 'drain': _stats(frames, waits_failed, slot_ms)}
    figure(f'hunt_{label}', row)
    d = row['drain']
    print(
        f'HUNT {label}: camera {d.get("camera_fps")} fps, host {d.get("host_fps")}, '
        f'slots {d.get("slots")}, id gaps {d.get("frame_id_gaps")}, '
        f'AFR {regs.get("AcquisitionFrameRate")}/{regs.get("AcquisitionFrameRate.max")}, '
        f'DLTL {regs.get("DeviceLinkThroughputLimit")}, current {regs.get("DeviceLinkCurrentThroughput")}, '
        f'temps {temps}',
        flush=True,
    )
    return row


def _set_afr(driver, step):
    node = _node(driver, 'AcquisitionFrameRate')
    top = node.Maximum()
    value = top if step == 'max' else top * step
    try:
        node.SetValue(value)
        return value
    except Exception as e:
        return f'{type(e).__name__}: {e}'


def _cycle_with_limit(driver, limit_bps):
    """Stop the camera (the host stream stays up), write the limit, start it again.

    The limit is read-only while the transport parameters are locked, so the
    camera is stopped and unlocked around the write; AcquisitionFrameRate is then
    put back to the maximum the new limit allows, as LumaViewPro's start does.
    """
    steps = {}

    def step(name, call):
        try:
            call()
            steps[name] = 'ok'
        except Exception as e:
            steps[name] = f'{type(e).__name__}: {e}'

    def stop():
        _node(driver, 'AcquisitionStop').Execute()
        _node(driver, 'AcquisitionStop').WaitUntilDone()

    def start():
        _node(driver, 'AcquisitionStart').Execute()
        _node(driver, 'AcquisitionStart').WaitUntilDone()

    node = _node(driver, 'DeviceLinkThroughputLimit')
    step('AcquisitionStop', stop)
    step('TLParamsLocked=0', lambda: _node(driver, 'TLParamsLocked').SetValue(0))
    target = node.Maximum() if limit_bps is None else limit_bps
    step(f'DeviceLinkThroughputLimit={target}', lambda: node.SetValue(target))
    step('AcquisitionFrameRate=max', lambda: _set_afr(driver, 'max'))
    step('TLParamsLocked=1', lambda: _node(driver, 'TLParamsLocked').SetValue(1))
    step('AcquisitionStart', start)
    return steps


def _slot_hunt(scope, dump_at_max):
    driver = scope._camera_driver
    cam = scope.camera
    rows = {}

    # 1. The frame rate backed off its maximum, at LumaViewPro's 300 MB/s limit.
    for step in AFR_STEPS:
        wrote = _set_afr(driver, step)
        print(f'HUNT AFR step {step}: wrote {wrote}', flush=True)
        rows[f'afr_{step}'] = {'step': step, 'wrote': wrote, **_point(scope, f'afr_{step}')}
    best = max(
        (k for k in rows if rows[k]['drain'].get('camera_fps')),
        key=lambda k: rows[k]['drain']['camera_fps'],
        default=None,
    )
    figure('hunt_afr_best', best)
    if best is not None:
        _set_afr(driver, rows[best]['step'])
        dump_best = _dump(driver)
        figure('hunt_nodes_changed_max_to_best_afr', _changed(dump_at_max, dump_best))
        _normal(scope, 'hunt_best_afr_live_pipeline')
        figure('hunt_best_afr_after_live_read', _registers(driver))
    _set_afr(driver, 'max')

    # 2. The throughput limit, stopped and written, at the maximum rate and backed off.
    for limit in DLTL_STEPS_BPS:
        steps = _cycle_with_limit(driver, limit)
        tag = 'max' if limit is None else f'{limit // 1_000_000}M'
        print(f'HUNT DLTL {tag}: {steps}', flush=True)
        rows[f'dltl_{tag}_afr_max'] = {'cycle': steps, **_point(scope, f'dltl_{tag}_afr_max')}
        rows[f'dltl_{tag}_afr_0.98'] = {
            'wrote': _set_afr(driver, 0.98),
            **_point(scope, f'dltl_{tag}_afr_0.98'),
        }
    figure('hunt_dltl_restore', _cycle_with_limit(driver, None))

    # 3. The other pixel format, the full-depth one.
    try:
        cam.set_image_mode('12bit_scientific')
        rows['mono12_afr_max'] = _point(scope, 'mono12_afr_max')
        _set_afr(driver, 0.98)
        rows['mono12_afr_0.98'] = _point(scope, 'mono12_afr_0.98')
    except Exception as e:
        figure('hunt_mono12_error', f'{type(e).__name__}: {e}')
    finally:
        cam.set_image_mode('8bit')

    # 4. The whole sensor at 2x2 (its widest frame is 1752), against the 950 crop.
    try:
        figure('hunt_full_sensor_frame', cam.set_frame_size(1752, 1752))
        rows['full_sensor_afr_max'] = _point(scope, 'full_sensor_afr_max')
        _set_afr(driver, 0.98)
        rows['full_sensor_afr_0.98'] = _point(scope, 'full_sensor_afr_0.98')
    except Exception as e:
        figure('hunt_full_sensor_error', f'{type(e).__name__}: {e}')
    finally:
        cam.set_frame_size(950, 950)
    _set_afr(driver, 'max')

    figure(
        'hunt_summary',
        {
            k: (
                v['drain'].get('camera_fps'),
                v['drain'].get('slots'),
                v['drain'].get('frame_id_gaps'),
            )
            for k, v in rows.items()
        },
    )


MARGIN_REPEATS = 3
MARGINS_2X2 = (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92)
MARGINS_1X1 = (1.0, 0.99, 0.98, 0.96)
MARGIN_MODES = (('mono10', '8bit'), ('mono12', '12bit_scientific'))


def _margin_run(scope):
    """At each binning and pixel format, the highest fraction of the maximum clean in every repeat.

    A fraction is clean when no repeat's drain skipped a slot and none lost a
    frame ID. The answer per configuration is the margin LumaViewPro's
    free-run setting needs there; a configuration clean at 1.0 needs none.
    """
    driver = scope._camera_driver
    cam = scope.camera
    answers, table = {}, {}
    for binning, size, margins in ((2, 950, MARGINS_2X2), (1, 1900, MARGINS_1X1)):
        cam.set_binning(binning)
        cam.set_frame_size(size, size)
        for mode_name, mode in MARGIN_MODES:
            config = f'{binning}x{binning}_{mode_name}'
            try:
                cam.set_image_mode(mode)
            except Exception as e:
                figure(f'margin_{config}_error', f'{type(e).__name__}: {e}')
                continue
            clean = []
            for fraction in margins:
                wrote = _set_afr(driver, 'max' if fraction == 1.0 else fraction)
                repeats = []
                for n in range(MARGIN_REPEATS):
                    row = _point(scope, f'margin_{config}_{fraction}_r{n + 1}', settle=n == 0)
                    repeats.append(row['drain'])
                skipped = [
                    sum(c for k, c in (d.get('slots') or {}).items() if k != 1) for d in repeats
                ]
                id_gaps = [d.get('frame_id_gaps') for d in repeats]
                table[f'{config}_{fraction}'] = {
                    'wrote': wrote,
                    'camera_fps': [d.get('camera_fps') for d in repeats],
                    'skipped': skipped,
                    'frame_id_gaps': id_gaps,
                }
                if all(s == 0 for s in skipped) and all(g == 0 for g in id_gaps):
                    clean.append(fraction)
            answers[config] = max(clean, default=None)
            print(f'MARGIN {config}: highest clean fraction {answers[config]}', flush=True)
    _set_afr(driver, 'max')
    cam.set_image_mode('8bit')
    cam.set_binning(2)
    cam.set_frame_size(950, 950)
    figure('margin_table', table)
    figure('margin_highest_clean_fraction', answers)
    check(
        'the margin run found a clean fraction at every configuration',
        bool(answers) and all(v is not None for v in answers.values()),
        str(answers),
    )


TARGETS_2X2 = (1.0, 0.99, 0.98, 0.96, 0.94)


def _target_run(scope):
    """The other way to set the rate: IDS's AcquisitionFrameRateTarget, at 2x2.

    The target is a rate the camera should reach if it can and not exceed
    (IDS manual, AcquisitionFrameRateTarget), where AcquisitionFrameRate is a
    rate it is told to run at; with the target enabled AcquisitionFrameRate
    is read-only. If the camera paces itself to a target at its maximum
    without skipping, the free-run fix is the target, not a margin.
    """
    driver = scope._camera_driver
    rate = _node(driver, 'AcquisitionFrameRate')
    _set_afr(driver, 'max')
    top = rate.Maximum()
    table = {}
    try:
        for fraction in TARGETS_2X2:
            steps = {}
            for name, value in (
                ('AcquisitionFrameRateTarget', top * fraction),
                ('AcquisitionFrameRateTargetEnable', True),
            ):
                try:
                    _node(driver, name).SetValue(value)
                    steps[name] = value
                except Exception as e:
                    steps[name] = f'{type(e).__name__}: {e}'
            repeats = [
                _point(scope, f'target_{fraction}_r{n + 1}', settle=n == 0)['drain']
                for n in range(MARGIN_REPEATS)
            ]
            table[f'target_{fraction}'] = {
                'wrote': steps,
                'camera_fps': [d.get('camera_fps') for d in repeats],
                'skipped': [
                    sum(c for k, c in (d.get('slots') or {}).items() if k != 1) for d in repeats
                ],
                'frame_id_gaps': [d.get('frame_id_gaps') for d in repeats],
            }
            print(f'TARGET {fraction}: {table[f"target_{fraction}"]}', flush=True)
    finally:
        try:
            _node(driver, 'AcquisitionFrameRateTargetEnable').SetValue(False)
        except Exception as e:
            figure('target_restore_error', f'{type(e).__name__}: {e}')
        _set_afr(driver, 'max')
    figure('target_table', table)


def _measure(session):
    scope = session.scope
    cam = scope.camera
    figure('camera_model', scope.capabilities.camera_model)
    figure('firmware', scope.diagnostics.run_pylon_diagnostic_probe(0).get('camera'))
    if not scope.imaging.is_streaming():
        scope.imaging.enable_streaming()
    check('the camera is streaming', scope.imaging.is_streaming())
    figure('exposure_ms', cam.set_exposure_ms(EXPOSURE_MS))
    cam.set_image_mode('8bit')

    cam.set_binning(2)
    _normal(scope, 'B1_binning_2')
    driver = scope._camera_driver
    has_nodes = hasattr(driver, 'remote_nodemap')
    before = _dump(driver) if has_nodes else None
    _drain(scope, 'B1_binning_2')
    if has_nodes:
        after = _dump(driver)
        figure('B1_binning_2_nodes', before)
        figure('B1_binning_2_changed_across_the_drain', _changed(before, after))
    _normal(scope, 'B1_binning_2_after')
    if has_nodes:
        _slot_hunt(scope, before)
        _margin_run(scope)
        _target_run(scope)
        _sweep(scope, before)
        _normal(scope, 'B1_binning_2_after_the_sweep')
        _exposure_sweep(scope)

    cam.set_binning(1)
    cam.set_frame_size(1900, 1900)
    _normal(scope, 'B2_1900')
    _drain(scope, 'B2_1900')


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_drain_rate')
    try:
        _measure(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
