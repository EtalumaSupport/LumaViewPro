#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bench-probe sweep orchestration for Pylon (Basler) cameras.

Iterates a configurable sweep matrix (image mode x frame size x
transport-specific knobs) and writes a per-cell JSON snapshot via
``scope.diagnostics.run_pylon_diagnostic_probe()`` -- the canonical
production probe API.

USB3 cells vary DeviceLinkThroughputLimit (DLTL).
GigE cells vary BandwidthReserveMode + GevSCPSPacketSize + GevSCPD.

The scope is the one connected to this machine, brought up the way every
host brings it up (``ScopeSession.create`` on the installation's own
``current.json``, then ``run_application_session``); nothing the sweep
sets is saved. Close LumaViewPro first: it holds the camera.

Transport: the camera's own nodes say it -- a GigE body has
``GevSCPSPacketSize``, a USB3 body ``DeviceLinkThroughputLimitMode``.

Usage:
    python -m tools.pylon_probe_sweep
    python -m tools.pylon_probe_sweep --duration 5.0
    python -m tools.pylon_probe_sweep --image-modes 8bit 12bit_scientific
    python -m tools.pylon_probe_sweep --resolutions 2100 sensor-max
    python -m tools.pylon_probe_sweep --dltl-modes Off On --dltl-values-mb 160 250 360
    python -m tools.pylon_probe_sweep --gige-bw-modes Default Performance \\
        --gige-packet-sizes 1500 9000

Outputs land in the log folder's ``camera_probe/``, one file per cell,
named for model + serial + firmware + host + dltl-token + timestamp.
Sweep runs print progress to stdout; one line per cell.
"""

import argparse
import contextlib
import logging
import platform
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import numpy as np

# This checkout's own API, whatever lumascope.pth the interpreter holds.
_LVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LVP_ROOT / 'src'))

from _lumascope.drivers.exceptions import HardwareError
from _lumascope.lvp_logger import logger
from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules.image_mode import ImageMode
from _lumascope.modules.path_utils import get_source_root
from _lumascope.modules.scope_session import ScopeSession


def _print_environment_header() -> None:
    """Log pypylon / Python / OS versions and every Pylon device the SDK sees.

    Runs before any connection attempt so the run is self-documenting even
    if connect() fails. Failure to enumerate (e.g. SDK runtime missing) is
    logged but does not abort -- _connect_camera surfaces the actual error.
    """
    try:
        pypylon_ver = _pkg_version('pypylon')
    except PackageNotFoundError:
        pypylon_ver = '<not installed via pip>'
    print(
        f'Environment: pypylon={pypylon_ver} '
        f'python={platform.python_version()} '
        f'os={platform.system()} {platform.release()} ({platform.machine()})'
    )
    try:
        from pypylon import pylon

        devs = pylon.TlFactory.GetInstance().EnumerateDevices()
        print(f'Pylon SDK enumerated {len(devs)} device(s):')
        for d in devs:
            print(
                f'  model={d.GetModelName()!r} '
                f'serial={d.GetSerialNumber()!r} '
                f'transport={d.GetDeviceClass()!r} '
                f'friendly={d.GetFriendlyName()!r}'
            )
    except Exception as e:
        print(f'WARNING: Pylon device enumeration raised {type(e).__name__}: {e}')


def _route_lvp_logging_to_stdout() -> None:
    """Wire LVP loggers to stdout so connect / cell-apply failures are diagnostic.

    The probe sweep is a CLI tool; without explicit handler attachment, log
    records from ``_lumascope.drivers.pyloncamera`` and ``_lumascope.modules.lumascope_api`` go
    nowhere. Without their output, ``connect() failed`` is opaque.
    """
    fmt = logging.Formatter('[%(levelname)s %(name)s] %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)
    # `LVP` is LumaViewPro's one logger root: the logger src/_lumascope/drivers/* import via
    # `from _lumascope.lvp_logger import logger`, and the parent of every `LVP.*` logger.
    log = logging.getLogger('LVP')
    log.setLevel(logging.DEBUG)
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in log.handlers
    ):
        log.addHandler(handler)


# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------


def _detect_transport(scope) -> str:
    """Return 'usb3', 'gige', or 'unknown' for the scope's camera.

    The camera's own nodes say it: a GigE body has ``GevSCPSPacketSize``
    (a GigE body has a throughput limit too, so that node is asked first),
    a USB3 body ``DeviceLinkThroughputLimitMode``. A camera with no named
    settings (not a Pylon camera) is 'unknown'.

    Raises:
        CameraReadFailedError: A read failed; a sweep that cannot tell its
            transport cannot pick its cells.
    """
    if 'device' not in scope.capabilities.camera_node_maps:
        return 'unknown'
    if scope.diagnostics.read_camera_node('GevSCPSPacketSize') is not None:
        return 'gige'
    if scope.diagnostics.read_camera_node('DeviceLinkThroughputLimitMode') is not None:
        return 'usb3'
    return 'unknown'


# ---------------------------------------------------------------------------
# Sweep cell construction
# ---------------------------------------------------------------------------


def _resolve_resolutions(spec: list[str], largest: tuple[int, int] | None):
    """Resolve --resolutions tokens to (w, h) tuples.

    Tokens:
      - 'sensor-max'   -> ``largest``, the largest frame the scope delivers
                          (``capabilities.camera_limits.frame_max``)
      - integer N      -> (N, N)
      - WxH            -> (W, H)

    A non-numeric, non-'sensor-max' token that doesn't contain 'x' indicates
    the operator typed a sibling flag name (e.g. 'dltl-modes') without the
    leading '--', so argparse's nargs='+' for --resolutions greedily consumed
    it. Print a clear error naming the bad token and exit cleanly rather than
    leave the operator a bare int() ValueError traceback. 'sensor-max' on a
    scope that knows no largest frame exits the same way: no frame stands in
    for it.
    """
    out = []
    for tok in spec:
        if tok == 'sensor-max':
            if largest is None:
                print(
                    'ERROR: --resolutions sensor-max, but this scope knows no largest frame '
                    '(capabilities.camera_limits.frame_max is None). Name the frame as WxH.',
                    file=sys.stderr,
                )
                sys.exit(2)
            out.append(largest)
            continue
        try:
            if 'x' in tok:
                w, h = tok.lower().split('x', 1)
                out.append((int(w), int(h)))
            else:
                n = int(tok)
                out.append((n, n))
        except ValueError:
            print(
                f'ERROR: --resolutions got non-numeric token {tok!r}. Expected '
                f"'sensor-max', an integer N (square N x N), or WxH (e.g. '2100x1500'). "
                f"Likely cause: a sibling flag was typed without '--' so argparse's "
                f'--resolutions list greedily consumed it. Re-run with the full --flag '
                f'prefix on every option.',
                file=sys.stderr,
            )
            sys.exit(2)
    return out


def _build_usb3_cells(args):
    """Cartesian product of USB3 sweep dimensions."""
    cells = []
    for image_mode in args.image_modes:
        for res in args.resolution_tuples:
            for mode in args.dltl_modes:
                if mode == 'Off':
                    cells.append(
                        {
                            'image_mode': image_mode,
                            'resolution': res,
                            'dltl_mode': 'Off',
                            'dltl_value': None,
                        }
                    )
                elif mode == 'On':
                    if not args.dltl_values_mb:
                        cells.append(
                            {
                                'image_mode': image_mode,
                                'resolution': res,
                                'dltl_mode': 'On',
                                'dltl_value': None,
                            }
                        )
                    else:
                        for v_mb in args.dltl_values_mb:
                            cells.append(
                                {
                                    'image_mode': image_mode,
                                    'resolution': res,
                                    'dltl_mode': 'On',
                                    'dltl_value': int(v_mb) * 1_000_000,
                                }
                            )
    return cells


def _build_gige_cells(args):
    """Cartesian product of GigE sweep dimensions."""
    cells = []
    for image_mode in args.image_modes:
        for res in args.resolution_tuples:
            for bw_mode in args.gige_bw_modes:
                for pkt in args.gige_packet_sizes:
                    for delay in args.gige_delays:
                        cells.append(
                            {
                                'image_mode': image_mode,
                                'resolution': res,
                                'bw_mode': bw_mode,
                                'packet_size': int(pkt),
                                'delay_ticks': int(delay),
                            }
                        )
    return cells


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def _apply_cell(scope, transport: str, cell: dict):
    """Apply one cell through the production setters. Returns a list of
    (knob, result) tuples for logging.

    The image mode and the frame are set as every host sets them
    (``camera.set_image_mode``, ``camera.set_frame_size``), each stopping
    and restarting the camera's grab around its write, so a cell measures
    the frames LVP itself takes.
    """
    log = []
    # A setting the camera refuses is one cell's data point, not a sweep
    # abort: record it and keep sweeping.
    image_mode = cell['image_mode']
    try:
        scope.camera.set_image_mode(ImageMode(image_mode))
        log.append(('image_mode', scope.camera.read_pixel_format()))
    except CameraSettingRejected as e:
        log.append(('image_mode', f'REJECTED: {e}'))

    w, h = cell['resolution']
    try:
        delivered = scope.camera.set_frame_size(w, h)
        log.append(('frame', f'{delivered.width}x{delivered.height}'))
    except CameraSettingRejected as e:
        log.append(('frame', f'REJECTED: {e}'))

    # Transport-specific knobs.
    # These setters are private (underscore-prefixed) on ImagingAPI
    # because they are bench-tooling artifacts, not part of the L2
    # contract. This tool is internal bench tooling; the underscore
    # access is intentional. A knob the camera refuses, does not offer or
    # does not answer is one cell's data point, recorded as the image
    # mode's is; no camera ends the sweep (not a HardwareError).
    if transport == 'usb3':
        knobs = [
            (
                'dltl',
                lambda: scope.imaging._set_device_link_throughput_limit(
                    mode=cell['dltl_mode'], value_bps=cell['dltl_value']
                ),
            )
        ]
    elif transport == 'gige':
        knobs = [
            ('bw_mode', lambda: scope.imaging._set_bandwidth_reserve_mode(cell['bw_mode'])),
            ('packet_size', lambda: scope.imaging._set_gev_packet_size(cell['packet_size'])),
            (
                'delay_ticks',
                lambda: scope.imaging._set_gev_inter_packet_delay(cell['delay_ticks']),
            ),
        ]
    else:
        knobs = []
    for name, set_knob in knobs:
        try:
            set_knob()
            log.append((name, 'written'))
        except HardwareError as e:
            log.append((name, f'FAILED: {e}'))

    return log


def _format_cell_id(transport: str, cell: dict) -> str:
    """Short human-readable cell identifier for stdout progress."""
    res = cell['resolution']
    parts = [
        cell['image_mode'],
        f'{res[0]}x{res[1]}',
    ]
    if transport == 'usb3':
        if cell['dltl_mode'] == 'Off':
            parts.append('dltl=Off')
        elif cell['dltl_value'] is not None:
            parts.append(f'dltl={cell["dltl_value"] // 1_000_000}M')
        else:
            parts.append('dltl=On(default)')
    elif transport == 'gige':
        parts.append(f'bw={cell["bw_mode"]}')
        parts.append(f'pkt={cell["packet_size"]}')
        parts.append(f'spcd={cell["delay_ticks"]}')
    return ' | '.join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _session_on_this_scope(serial_filter: str | None):
    """The scope connected to this machine, brought up as every host brings it up.

    The settings are the installation's own ``current.json``; a root
    without one is refused, because the shipped template would configure a
    scope that is not the one on the bench. Nothing the sweep sets is
    saved (no settings file is given). A scope with no motor board comes
    straight up. The session is shut down on leaving, however the sweep
    ends.
    """
    root = get_source_root()
    if not (root / 'data' / 'current.json').is_file():
        print(
            f'ERROR: no {root / "data" / "current.json"} -- the session would come up on the '
            'shipped template, not this installation. Start LumaViewPro once here first.'
        )
        sys.exit(2)
    session = ScopeSession.create(
        ScopeSession.load_user_settings(str(root)),
        source_path=str(root),
        warn_pre_release=False,
    )
    try:
        session.run_application_session()
        serial = session.scope.capabilities.camera_serial_number
        if serial_filter and serial != serial_filter:
            print(
                f'ERROR: Connected camera serial {serial!r} does not '
                f'match --camera-serial {serial_filter!r}'
            )
            sys.exit(1)
        yield session
    finally:
        session.shutdown()


def _host_load_loop(stop_event: threading.Event) -> None:
    """Saturate one CPU core with the kind of per-frame work LVP does on the
    host -- np.std over a 2100x2100 frame -- until stopped.

    The bench question is why LVP sees payload-discard errors when the Pylon
    Viewer bandwidth test (same ROI) does not: the tester does no host-side
    work, so it never starves the grab worker. On a slow host, LVP's live
    display + per-frame processing + AF SetValue saturate the cores and the
    grab worker cannot drain the camera's buffers fast enough, overflowing
    the camera-side FIFO (the "payload discarded / bandwidth insufficient"
    error). This load reproduces that starvation so the sweep can show
    whether a higher MaxNumBuffer absorbs it. std is chosen deliberately --
    it is the costliest per-frame stat (two passes + sqrt) and mirrors the
    engineering-preview compute.
    """
    frame = np.random.randint(0, 4096, size=(2100, 2100), dtype=np.uint16)
    while not stop_event.is_set():
        float(np.std(frame))


class _HostLoad:
    """Manage N background CPU workers that simulate slow-host load."""

    def __init__(self, workers: int):
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=_host_load_loop,
                args=(self._stop,),
                name=f'host-load-{i}',
                daemon=True,
            )
            for i in range(max(0, workers))
        ]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(
        description='Pylon (Basler) bench-probe sweep orchestrator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--duration', type=float, default=3.0, help='Diagnostic-probe sampling window (seconds).'
    )
    parser.add_argument(
        '--settle', type=float, default=1.0, help='Seconds to wait between cell setup and probe.'
    )
    parser.add_argument(
        '--camera-serial',
        default=None,
        help='If set, abort unless the connected camera has this serial number.',
    )
    parser.add_argument(
        '--image-modes',
        nargs='+',
        default=[ImageMode.EIGHT_BIT.value],
        choices=[m.value for m in ImageMode],
        help='Image modes to sweep; each sets the pixel format LVP captures it in '
        '(8bit -> Mono8, the 12-bit modes -> Mono12).',
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        default=['2100'],
        help='Frame sizes to sweep, as LVP frames them. Tokens: integer N -> '
        '(N,N); WxH -> (W,H); sensor-max -> the largest frame the scope '
        'delivers.',
    )
    # USB3 sweep
    parser.add_argument(
        '--dltl-modes',
        nargs='+',
        default=['On'],
        choices=['On', 'Off'],
        help='DLTL modes to sweep (USB3 cells).',
    )
    parser.add_argument(
        '--dltl-values-mb',
        nargs='*',
        type=int,
        default=[],
        help='DLTL values in MB/s (USB3 cells, used when '
        'mode=On). Empty list means use the camera '
        'default.',
    )
    # GigE sweep
    parser.add_argument(
        '--gige-bw-modes',
        nargs='+',
        default=['Performance'],
        choices=['Default', 'Performance'],
        help='BandwidthReserveMode values (GigE cells).',
    )
    parser.add_argument(
        '--gige-packet-sizes',
        nargs='+',
        type=int,
        default=[1500],
        help='GevSCPSPacketSize values (GigE cells).',
    )
    parser.add_argument(
        '--host-load-workers',
        type=int,
        default=0,
        help='Background CPU workers running np.std on a 2100x2100 frame '
        'during the sweep, simulating the slow-host per-frame load that '
        'starves the Pylon grab worker and overflows the camera FIFO. '
        '0 = none (transport-only baseline, like the Pylon Viewer bandwidth '
        'test). To test the camera-reset hypothesis, power-cycle '
        'the mainboard between runs and compare the failed-buffer delta.',
    )
    parser.add_argument(
        '--gige-delays',
        nargs='+',
        type=int,
        default=[0],
        help='GevSCPD inter-packet delay ticks (GigE cells).',
    )
    args = parser.parse_args()

    _route_lvp_logging_to_stdout()
    _print_environment_header()

    # Record the invocation through the production logger so it lands in the
    # LVP_Log bundle. The tool's own progress + deltas go to stdout only; a
    # support bundle therefore cannot tell which command produced a run, and
    # several runs append to the same log file. This line anchors each run and
    # captures the knobs that are not otherwise logged (host_load_workers,
    # duration).
    logger.info(f'[PROBE Sweep] run start: {" ".join(sys.argv)}')
    logger.info(
        f'[PROBE Sweep] params: duration={args.duration}s settle={args.settle}s '
        f'image_modes={args.image_modes} resolutions={args.resolutions} '
        f'host_load_workers={args.host_load_workers} '
        f'camera_serial={args.camera_serial}'
    )

    with _session_on_this_scope(args.camera_serial) as session:
        _sweep(session.scope, args)


def _sweep(scope, args) -> None:
    """Build the cells for the scope's transport and probe each in turn."""
    transport = _detect_transport(scope)
    limits = scope.capabilities.camera_limits
    largest = None
    if limits is not None and limits.frame_max is not None:
        largest = (limits.frame_max.width, limits.frame_max.height)

    print(
        f'Camera: model={scope.capabilities.camera_model!r} '
        f'serial={scope.capabilities.camera_serial_number!r} transport={transport!r} '
        f'largest frame={largest}'
    )

    args.resolution_tuples = _resolve_resolutions(args.resolutions, largest)

    if transport == 'usb3':
        cells = _build_usb3_cells(args)
    elif transport == 'gige':
        cells = _build_gige_cells(args)
    else:
        print(f'ERROR: unknown transport {transport!r} -- cannot build sweep')
        sys.exit(2)

    print(f'Sweep: {len(cells)} cells ({transport.upper()})')

    host_load = _HostLoad(args.host_load_workers)
    if args.host_load_workers > 0:
        print(
            f'Host load: {args.host_load_workers} CPU worker(s) running np.std on a '
            f'2100x2100 frame (simulating slow-host per-frame work that starves the '
            f'grab worker; failed= delta below is the payload-discard count)'
        )
        host_load.start()

    # The session's camera is grabbing once it is up, so the stats counters
    # advance through each probe's window.
    try:
        for i, cell in enumerate(cells, 1):
            cell_id = _format_cell_id(transport, cell)
            print(f'[{i}/{len(cells)}] {cell_id}')
            apply_log = _apply_cell(scope, transport, cell)
            for knob, status in apply_log:
                if status is not True:
                    print(f'    apply {knob}: {status}')
            if args.settle > 0:
                time.sleep(args.settle)
            snapshot = scope.diagnostics.run_pylon_diagnostic_probe(duration_s=args.duration)
            out_path = snapshot.get('output_path')
            errors = snapshot.get('errors') or []
            deltas = snapshot.get('deltas') or {}
            delta_summary = (
                f'total={deltas.get("Statistic_Total_Buffer_Count")} '
                f'failed={deltas.get("Statistic_Failed_Buffer_Count")} '
                f'resync={deltas.get("Statistic_Resynchronization_Count")} '
                f'missed={deltas.get("Statistic_Missed_Frame_Count")}'
            )
            print(f'    deltas: {delta_summary}')
            print(f'    -> {out_path}' + (f' (errors: {errors})' if errors else ''))
            # Also log the result so the failed-buffer count is in the bundle,
            # keyed to the cell + the load knob for this run.
            logger.info(
                f'[PROBE Sweep] cell {i}/{len(cells)} {cell_id} '
                f'host_load_workers={args.host_load_workers}: {delta_summary}'
            )
    finally:
        host_load.stop()


if __name__ == '__main__':
    main()
