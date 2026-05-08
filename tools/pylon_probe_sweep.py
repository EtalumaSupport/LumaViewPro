#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bench-probe sweep orchestration for Pylon (Basler) cameras.

Iterates a configurable sweep matrix (pixel format x resolution x
transport-specific knobs) and writes a per-cell JSON snapshot via
``Lumascope.run_pylon_diagnostic_probe()`` -- the canonical
production probe API.

USB3 cells vary DeviceLinkThroughputLimit (DLTL).
GigE cells vary BandwidthReserveMode + GevSCPSPacketSize + GevSCPD.

Transport detection: ``DeviceTransportLayerType`` from the camera's
TL nodemap (USB3 -> ``BaslerUsb3Vision``, GigE -> ``BaslerGigE``).
Falls back to model substring (``gm`` -> GigE; otherwise USB3) if
the node is not exposed.

Usage:
    python -m tools.pylon_probe_sweep
    python -m tools.pylon_probe_sweep --duration 5.0
    python -m tools.pylon_probe_sweep --pixel-formats Mono8 Mono12
    python -m tools.pylon_probe_sweep --resolutions 2100 sensor-max
    python -m tools.pylon_probe_sweep --dltl-modes Off On --dltl-values-mb 160 250 360
    python -m tools.pylon_probe_sweep --gige-bw-modes Default Performance \\
        --gige-packet-sizes 1500 9000

Outputs land in ``data/pylon_probe/`` with per-cell filenames keyed
on model + serial + firmware + host + dltl-token + timestamp. Sweep
runs print progress to stdout; one line per cell.

Production-aligned per Architecture Rule 22: imports the canonical
``PylonCamera`` driver and ``Lumascope.run_pylon_diagnostic_probe()``
API method. Transport-specific setters
(``set_device_link_throughput_limit``,
``set_bandwidth_reserve_mode``, ``set_gev_packet_size``,
``set_gev_inter_packet_delay``) are called per cell. No /tmp scripts;
no ad-hoc subprocess invocations.
"""

import argparse
import logging
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

# Add LumaViewPro root to path so imports work when run as script
_LVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LVP_ROOT))

from drivers.pyloncamera import PylonCamera
from modules.lumascope_api import Lumascope


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
    print(f'Environment: pypylon={pypylon_ver} '
          f'python={platform.python_version()} '
          f'os={platform.system()} {platform.release()} ({platform.machine()})')
    try:
        from pypylon import pylon
        devs = pylon.TlFactory.GetInstance().EnumerateDevices()
        print(f'Pylon SDK enumerated {len(devs)} device(s):')
        for d in devs:
            print(f'  model={d.GetModelName()!r} '
                  f'serial={d.GetSerialNumber()!r} '
                  f'transport={d.GetDeviceClass()!r} '
                  f'friendly={d.GetFriendlyName()!r}')
    except Exception as e:
        print(f'WARNING: Pylon device enumeration raised {type(e).__name__}: {e}')


def _route_lvp_logging_to_stdout() -> None:
    """Wire LVP loggers to stdout so connect / cell-apply failures are diagnostic.

    The probe sweep is a CLI tool; without explicit handler attachment, log
    records from ``drivers.pyloncamera`` and ``modules.lumascope_api`` go
    nowhere. Without their output, ``connect() failed`` is opaque.
    """
    fmt = logging.Formatter('[%(levelname)s %(name)s] %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)
    # `lvp_logger` is the logger drivers/* import via `from lvp_logger import logger`.
    # `LVP` is the parent for `LVP.camera` / `LVP.serial` / `LVP.autofocus`.
    for name in ('lvp_logger', 'LVP'):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
                   for h in log.handlers):
            log.addHandler(handler)


# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------

def _detect_transport(camera) -> str:
    """Return 'usb3', 'gige', or 'unknown' for a connected PylonCamera.

    Reads DeviceTransportLayerType from the camera's TL nodemap. Falls
    back to model substring detection if the node is not exposed.
    """
    if not camera.active:
        return 'unknown'
    try:
        tl_type = camera.active.GetTLNodeMap().GetNode(
            'DeviceTransportLayerType')
        if tl_type is not None:
            v = tl_type.GetValue()
            if 'Usb' in v or 'USB' in v or 'usb' in v:
                return 'usb3'
            if 'GigE' in v or 'Gige' in v or 'GIGE' in v:
                return 'gige'
    except Exception:
        pass

    # Fallback: model-string substring match.
    model = (getattr(camera, 'model_name', None) or '').lower()
    # Basler USB3 model suffixes: 'um' (mono USB), 'uc' (color USB)
    # Basler GigE model suffixes: 'gm' (mono GigE), 'gc' (color GigE)
    # Use the trailing position so 'um' inside 'gummed' etc. doesn't match
    # (Basler models follow strict naming conventions; checking the last
    # 4 chars covers the documented family suffixes).
    tail = model[-4:]
    if 'gm' in tail or 'gc' in tail:
        return 'gige'
    if 'um' in tail or 'uc' in tail:
        return 'usb3'
    return 'unknown'


def _sensor_max_resolution(camera) -> tuple[int, int]:
    """Read sensor-native (Width.Max, Height.Max) from the camera."""
    try:
        w = int(camera.active.Width.GetMax())
        h = int(camera.active.Height.GetMax())
        return (w, h)
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Sweep cell construction
# ---------------------------------------------------------------------------

def _resolve_resolutions(spec: list[str], sensor_w: int, sensor_h: int):
    """Resolve --resolutions tokens to (w, h) tuples.

    Tokens:
      - 'sensor-max'   -> (sensor_w, sensor_h)
      - integer N      -> (N, N)
      - WxH            -> (W, H)
    """
    out = []
    for tok in spec:
        if tok == 'sensor-max':
            out.append((sensor_w, sensor_h))
        elif 'x' in tok:
            w, h = tok.lower().split('x', 1)
            out.append((int(w), int(h)))
        else:
            n = int(tok)
            out.append((n, n))
    return out


def _build_usb3_cells(args):
    """Cartesian product of USB3 sweep dimensions."""
    cells = []
    for pf in args.pixel_formats:
        for res in args.resolution_tuples:
            for mode in args.dltl_modes:
                if mode == 'Off':
                    cells.append(dict(pixel_format=pf, resolution=res,
                                      dltl_mode='Off', dltl_value=None))
                elif mode == 'On':
                    if not args.dltl_values_mb:
                        cells.append(dict(pixel_format=pf, resolution=res,
                                          dltl_mode='On', dltl_value=None))
                    else:
                        for v_mb in args.dltl_values_mb:
                            cells.append(dict(pixel_format=pf, resolution=res,
                                              dltl_mode='On',
                                              dltl_value=int(v_mb) * 1_000_000))
    return cells


def _build_gige_cells(args):
    """Cartesian product of GigE sweep dimensions."""
    cells = []
    for pf in args.pixel_formats:
        for res in args.resolution_tuples:
            for bw_mode in args.gige_bw_modes:
                for pkt in args.gige_packet_sizes:
                    for delay in args.gige_delays:
                        cells.append(dict(
                            pixel_format=pf, resolution=res,
                            bw_mode=bw_mode, packet_size=int(pkt),
                            delay_ticks=int(delay),
                        ))
    return cells


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

def _apply_cell(scope, transport: str, cell: dict, sensor_w: int, sensor_h: int):
    """Apply per-cell config via the production setters. Returns a list
    of (knob, ok) tuples for logging.

    All writes happen inside ``camera.update_camera_config()`` so the grab
    loop is stopped before geometry / ROI / transport knobs are written.
    pypylon 26.4.1 enforces RO on Width / Height / OffsetX / OffsetY /
    MaxNumBuffer while grabbing; without this wrapping the writes raise
    ``AccessException: Node is not writable`` and the cell runs against
    the camera's previous-cell config.
    """
    log = []
    with scope.camera.update_camera_config():
        # Pixel format
        pf = cell['pixel_format']
        log.append(('pixel_format', scope.set_pixel_format(pf)))

        # Resolution + centered ROI
        w, h = cell['resolution']
        if w > 0 and h > 0:
            # Use direct camera Width/Height/OffsetX/OffsetY for the
            # centered ROI -- set_frame_size does not support centering.
            try:
                cam = scope.camera.active
                cam.OffsetX.SetValue(0)
                cam.OffsetY.SetValue(0)
                cam.Width.SetValue(w)
                cam.Height.SetValue(h)
                ox = max(0, (sensor_w - w) // 2) if sensor_w >= w else 0
                oy = max(0, (sensor_h - h) // 2) if sensor_h >= h else 0
                ox -= ox % int(cam.OffsetX.GetInc()) if cam.OffsetX.GetInc() else 0
                oy -= oy % int(cam.OffsetY.GetInc()) if cam.OffsetY.GetInc() else 0
                cam.OffsetX.SetValue(ox)
                cam.OffsetY.SetValue(oy)
                log.append(('resolution', True))
            except Exception as e:
                log.append(('resolution', f'FAILED: {e}'))

        # Transport-specific knobs
        if transport == 'usb3':
            ok = scope.set_device_link_throughput_limit(
                mode=cell['dltl_mode'], value_bps=cell['dltl_value'])
            log.append(('dltl', ok))
        elif transport == 'gige':
            log.append(('bw_mode', scope.set_bandwidth_reserve_mode(cell['bw_mode'])))
            log.append(('packet_size', scope.set_gev_packet_size(cell['packet_size'])))
            log.append(('delay_ticks', scope.set_gev_inter_packet_delay(cell['delay_ticks'])))

    return log


def _format_cell_id(transport: str, cell: dict) -> str:
    """Short human-readable cell identifier for stdout progress."""
    res = cell['resolution']
    parts = [
        cell['pixel_format'],
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

def _connect_camera(serial_filter: str | None) -> PylonCamera:
    """Return a connected PylonCamera; abort on failure with a diagnostic message.

    ``Camera.__init__`` auto-connects per the driver-registry contract, so
    constructing the PylonCamera IS the connect. ``cam.found`` is the
    base-class success signal. Calling ``cam.connect()`` again would
    re-Open the same device and hit "Device is exclusively opened".
    """
    cam = PylonCamera()
    if not getattr(cam, 'found', False):
        print('ERROR: PylonCamera connect failed. See [ERROR] log lines '
              'above for the SDK reason.')
        sys.exit(1)
    if serial_filter:
        try:
            actual = cam.active.GetDeviceInfo().GetSerialNumber()
        except Exception:
            actual = None
        if actual != serial_filter:
            print(f'ERROR: Connected camera serial {actual!r} does not '
                  f'match --camera-serial {serial_filter!r}')
            cam.disconnect()
            sys.exit(1)
    return cam


def _make_minimal_scope(camera: PylonCamera) -> Lumascope:
    """Construct a minimal Lumascope shell with only the camera attached.

    Bypasses the full Lumascope.__init__ (which expects scope / board /
    settings). The setters this tool calls (set_pixel_format,
    set_frame_size, set_device_link_throughput_limit, GigE setters,
    run_pylon_diagnostic_probe) reach for both ``self.camera`` and
    ``self._camera_cache_lock``; we initialize the lock explicitly so the
    shortcut is safe.
    """
    import threading
    scope = Lumascope.__new__(Lumascope)
    scope.camera = camera
    scope._camera_cache_lock = threading.Lock()
    scope._camera_cache = {
        'active': True,
        'gain': 0.0,
        'exposure_ms': 0.0,
        'frame_size': {'width': 0, 'height': 0},
        'max_frame_size': {'width': 0, 'height': 0},
        'min_frame_size': {'width': 0, 'height': 0},
        'max_exposure': 0.0,
        'max_gain': 0.0,
        'pixel_format': None,
        'binning': 1,
    }
    return scope


def main():
    parser = argparse.ArgumentParser(
        description='Pylon (Basler) bench-probe sweep orchestrator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Diagnostic-probe sampling window (seconds).')
    parser.add_argument('--settle', type=float, default=1.0,
                        help='Seconds to wait between cell setup and probe.')
    parser.add_argument('--camera-serial', default=None,
                        help='If set, abort unless the connected camera '
                             'has this serial number.')
    parser.add_argument('--pixel-formats', nargs='+',
                        default=['Mono8'],
                        help='Pixel formats to sweep (Mono8, Mono12, ...).')
    parser.add_argument('--resolutions', nargs='+',
                        default=['2100'],
                        help='Resolutions to sweep. Tokens: integer N -> '
                             '(N,N); WxH -> (W,H); sensor-max -> camera '
                             'sensor maximum.')
    # USB3 sweep
    parser.add_argument('--dltl-modes', nargs='+',
                        default=['On'],
                        choices=['On', 'Off'],
                        help='DLTL modes to sweep (USB3 cells).')
    parser.add_argument('--dltl-values-mb', nargs='*', type=int,
                        default=[],
                        help='DLTL values in MB/s (USB3 cells, used when '
                             'mode=On). Empty list means use the camera '
                             'default.')
    # GigE sweep
    parser.add_argument('--gige-bw-modes', nargs='+',
                        default=['Performance'],
                        choices=['Default', 'Performance'],
                        help='BandwidthReserveMode values (GigE cells).')
    parser.add_argument('--gige-packet-sizes', nargs='+', type=int,
                        default=[1500],
                        help='GevSCPSPacketSize values (GigE cells).')
    parser.add_argument('--gige-delays', nargs='+', type=int,
                        default=[0],
                        help='GevSCPD inter-packet delay ticks (GigE cells).')
    args = parser.parse_args()

    _route_lvp_logging_to_stdout()
    _print_environment_header()

    camera = _connect_camera(args.camera_serial)
    scope = _make_minimal_scope(camera)

    transport = _detect_transport(camera)
    sensor_w, sensor_h = _sensor_max_resolution(camera)

    print(f'Camera: model={camera.model_name!r} transport={transport!r} '
          f'sensor={sensor_w}x{sensor_h}')

    # Resolve resolution tokens against actual sensor size
    args.resolution_tuples = _resolve_resolutions(
        args.resolutions, sensor_w, sensor_h)

    if transport == 'usb3':
        cells = _build_usb3_cells(args)
    elif transport == 'gige':
        cells = _build_gige_cells(args)
    else:
        print(f'ERROR: unknown transport {transport!r} -- cannot build sweep')
        camera.disconnect()
        sys.exit(2)

    print(f'Sweep: {len(cells)} cells ({transport.upper()})')

    # Camera must be grabbing for stats counters to advance.
    camera.start_grabbing()
    try:
        for i, cell in enumerate(cells, 1):
            cell_id = _format_cell_id(transport, cell)
            print(f'[{i}/{len(cells)}] {cell_id}')
            apply_log = _apply_cell(scope, transport, cell, sensor_w, sensor_h)
            for knob, status in apply_log:
                if status is not True:
                    print(f'    apply {knob}: {status}')
            if args.settle > 0:
                time.sleep(args.settle)
            snapshot = scope.run_pylon_diagnostic_probe(
                duration_s=args.duration)
            out_path = snapshot.get('output_path')
            errors = snapshot.get('errors') or []
            print(f'    -> {out_path}'
                  + (f' (errors: {errors})' if errors else ''))
    finally:
        camera.stop_grabbing()
        camera.disconnect()


if __name__ == '__main__':
    main()
