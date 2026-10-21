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

    python tests/capability/camera_p11_drain_rate.py --hardware  # the connected IDS scope

The drain, the dump and the sweep reach the driver directly, as the unpack
benchmark does: they are a bench instrument, not API members. While a drain
runs the live view stores no frame, so the stream check may report one
stall; that report is expected. The stream's transfer settings are restored
at the end. The simulated camera has no drain, so the simulator run is a
void. The checks, the figures and the dumps are in the run's verdict file
(the harness's THE RECORD).
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void

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
