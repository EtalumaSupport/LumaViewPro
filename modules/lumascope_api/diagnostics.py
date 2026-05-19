# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""DiagnosticsAPI -- sub-API for hardware diagnostic probes.

Wave 7 Phase 5 decomposition. Stateless probes for camera / motor /
LED hardware. Bodies live here; Lumascope keeps thin one-line
forwarders for the 7 public methods until 5f retires them.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.4 for the canonical
method list. No persistent state -- per-call probes.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np

from lvp_logger import logger

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope


class DiagnosticsAPI:
    """Diagnostics sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope') -> None:
        self._scope = scope

    # --- Camera probes ---
    def get_camera_temperatures(self) -> dict:
        """Get all camera temperature sensor readings.

        Returns:
            dict: Mapping of sensor name to temperature in degC.
            Empty dict if camera is inactive or has no temperature sensors.
        """
        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            return {}
        try:
            return self._scope._camera_driver.get_all_temperatures()
        except Exception as e:
            logger.debug(f'[SCOPE API ] get_camera_temperatures failed: {e}')
            return {}

    def get_camera_diagnostic_info(self) -> dict:
        """Read-only snapshot of camera state for diagnostics.

        Returns the values that ``modules/tech_support_report.py`` and
        bench tools used to read directly off the driver. Each field is
        independently guarded so partial driver support yields a partial
        dict rather than an exception.

        Returns:
            dict: Camera diagnostic snapshot. Keys may include
                'model', 'resolution', 'pixel_format', 'gain', 'exposure_ms',
                'max_gain', 'max_exposure_ms', 'temperatures', plus per-key
                error strings for fields the driver couldn't supply.
                Returns ``{'connected': False}`` if the camera is inactive.
        """
        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            return {'connected': False}

        info: dict = {'connected': True}

        def _try(key, fn):
            try:
                info[key] = fn()
            except Exception as e:
                info[key] = f'Error: {e}'

        _try('model', lambda: self._scope._camera_driver.get_model_name())
        _try('pixel_format', lambda: self._scope._camera_driver.get_pixel_format())

        try:
            fs = self._scope._camera_driver.get_frame_size()
            info['resolution'] = f"{fs.get('width', '?')}x{fs.get('height', '?')}"
            info['frame_size'] = fs
        except Exception as e:
            info['resolution'] = f'Error: {e}'

        _try('gain', lambda: self._scope.imaging.get_gain())
        _try('exposure_ms', lambda: self._scope.imaging.get_exposure_time())
        _try('max_gain', lambda: self._scope._camera_driver.get_max_gain())
        _try('max_exposure_ms', lambda: self._scope._camera_driver.get_max_exposure())

        info['temperatures'] = self.get_camera_temperatures()
        return info

    def run_camera_bandwidth_test(
        self,
        num_frames: int,
        *,
        timeout_s: float = 60.0,
        progress_cb=None,
    ) -> dict:
        """Run an N-frame camera throughput test through the production capture path.

        Routes every frame grab through ``Lumascope.get_image()`` so the
        bandwidth numbers reflect what protocol/preview capture actually
        sees. Bypassing this method (calling ``self._scope._camera_driver.get_image()``
        directly) is a Rule-1 layer violation and the resulting numbers
        are not comparable to production capture.

        Args:
            num_frames: Total frames to grab.
            timeout_s: Hard wall-clock cutoff in seconds; the test stops
                early and marks ``passed=False`` if exceeded.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 250 frames.

        Returns:
            dict: Same shape as the legacy ``CameraBandwidthTest.run()`` --
                num_frames_requested, num_frames_received, num_frames_none,
                num_frames_error, total_bytes, elapsed_seconds,
                mb_per_second, fps_actual, frame_sizes, errors, passed.
        """
        results = {
            'num_frames_requested': int(num_frames),
            'num_frames_received': 0,
            'num_frames_none': 0,
            'num_frames_error': 0,
            'total_bytes': 0,
            'elapsed_seconds': 0,
            'mb_per_second': 0.0,
            'fps_actual': 0.0,
            'frame_sizes': [],
            'errors': [],
            'passed': True,
        }

        # Annotate with current camera state -- same fields the legacy
        # tech-support test attached to its result dict.
        cam_info = self.get_camera_diagnostic_info()
        if cam_info.get('connected'):
            for key in ('resolution', 'pixel_format'):
                if key in cam_info:
                    results[key] = cam_info[key]

        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            results['passed'] = False
            results['errors'].append('Camera not active')
            return results

        frame_size_set = set()
        start = time.monotonic()
        for i in range(int(num_frames)):
            if progress_cb and i % 250 == 0:
                try:
                    progress_cb(int(100 * i / max(num_frames, 1)),
                                f"Frame {i}/{num_frames}")
                except Exception:
                    pass
            try:
                # force_to_8bit=False keeps native depth so frame size
                # reflects the actual bytes the SDK delivered.
                frame = self._scope.imaging.get_image(force_to_8bit=False, force_new_capture=True)
                if frame is None or frame is False:
                    results['num_frames_none'] += 1
                else:
                    results['num_frames_received'] += 1
                    nbytes = getattr(frame, 'nbytes', None) or len(frame)
                    results['total_bytes'] += nbytes
                    frame_size_set.add(int(nbytes))
            except Exception as e:
                results['num_frames_error'] += 1
                if len(results['errors']) < 20:
                    results['errors'].append(
                        f"Frame {i}: {type(e).__name__}: {e}")

            if time.monotonic() - start > timeout_s:
                results['errors'].append(
                    f"Timeout at frame {i} after {timeout_s}s")
                results['passed'] = False
                break

        elapsed = time.monotonic() - start
        results['elapsed_seconds'] = round(elapsed, 2)
        if elapsed > 0:
            results['mb_per_second'] = round(
                results['total_bytes'] / (1024 * 1024) / elapsed, 2)
            results['fps_actual'] = round(
                results['num_frames_received'] / elapsed, 1)
        results['frame_sizes'] = sorted(frame_size_set)

        if results['num_frames_none'] > 0:
            results['passed'] = False
            results['errors'].append(
                f"{results['num_frames_none']} frames returned None -- "
                f"possible USB disconnect or bandwidth issue")
        if results['num_frames_error'] > 0:
            results['passed'] = False
        if len(frame_size_set) > 1:
            results['passed'] = False
            results['errors'].append(
                f"Inconsistent frame sizes: {sorted(frame_size_set)} -- "
                f"possible data corruption or config change during test")

        logger.info(
            f"[SCOPE API ] run_camera_bandwidth_test: {results['num_frames_received']}/{num_frames} "
            f"frames in {results['elapsed_seconds']}s "
            f"({results['mb_per_second']} MB/s, {results['fps_actual']} fps), "
            f"passed={results['passed']}"
        )
        return results

    def run_grab_lifecycle_benchmark(
        self,
        num_cycles: int = 100,
        inter_cycle_delay_ms: float = 0.0,
        vary_settings: bool = False,
        *,
        slow_threshold_s: float = 3.0,
        progress_cb=None,
    ) -> dict:
        """Characterize stop_grabbing/start_grabbing latency under back-to-back cycling.

        CAM-1 step (0a) -- empirical floor for the SDK's "minimum safe
        interval between StopGrabbing and the next StartGrabbing" instead
        of relying on Basler-published numbers. Typical case is 130-150 ms;
        the pathological ~11 s case has been observed when StopGrabbing
        fires within ~275 ms of a prior StartGrabbing before the camera
        produces a frame. Sweeping ``inter_cycle_delay_ms`` through
        0/50/100/200/500/1000 ms across runs reveals the smallest delay
        that yields ZERO slow cycles.

        Stays inside the API: drops to ``self._scope._camera_driver.stop_grabbing`` /
        ``start_grabbing`` directly, which is a Rule-1 downward call from
        the API into its driver -- same pattern as ``set_frame_size`` etc.

        Args:
            num_cycles: Stop/start cycles to perform.
            inter_cycle_delay_ms: Sleep between StopGrabbing and the next
                StartGrabbing (and any settings churn).
            vary_settings: When True, alternate gain (1.0 <-> 4.0) and
                exposure (10 ms <-> 50 ms) between cycles to reproduce the
                per-step protocol pattern that caused STALL-1.
            slow_threshold_s: Cycle wall-time considered "slow" -- counted
                separately so the operator sees how often the pathological
                case fires under the chosen delay.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 10 cycles.

        Returns:
            dict with: num_cycles, inter_cycle_delay_ms, vary_settings,
                slow_threshold_s, slow_cycle_count, slow_cycles (list of
                {idx, cycle_s, stop_s, start_s}), cycle_p50/p95/p99,
                stop_p50/p95/p99, start_p50/p95/p99, total_elapsed_s,
                camera_model, pylon_version, errors, written_to.
        """
        results = {
            'num_cycles': int(num_cycles),
            'inter_cycle_delay_ms': float(inter_cycle_delay_ms),
            'vary_settings': bool(vary_settings),
            'slow_threshold_s': float(slow_threshold_s),
            'slow_cycle_count': 0,
            'slow_cycles': [],
            'cycle_p50_s': 0.0, 'cycle_p95_s': 0.0, 'cycle_p99_s': 0.0,
            'stop_p50_s': 0.0,  'stop_p95_s': 0.0,  'stop_p99_s': 0.0,
            'start_p50_s': 0.0, 'start_p95_s': 0.0, 'start_p99_s': 0.0,
            'total_elapsed_s': 0.0,
            'camera_model': None,
            'pylon_version': None,
            'errors': [],
            'written_to': None,
        }

        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            results['errors'].append('Camera not active')
            return results

        cam_info = self.get_camera_diagnostic_info()
        results['camera_model'] = cam_info.get('model')
        results['pylon_version'] = cam_info.get('sdk_version') or cam_info.get('pylon_version')

        cycle_times, stop_times, start_times = [], [], []
        delay_s = max(0.0, float(inter_cycle_delay_ms) / 1000.0)

        # Snapshot current settings so we can restore even when vary_settings
        # is on -- the benchmark must not leave the camera in an arbitrary state.
        original_gain = getattr(self._scope._camera_driver, 'gain', None)
        original_exposure = getattr(self._scope._camera_driver, 'exposure_time', None)

        t_overall_start = time.monotonic()
        for i in range(int(num_cycles)):
            if progress_cb and i % 10 == 0:
                try:
                    progress_cb(int(100 * i / max(num_cycles, 1)),
                                f"Cycle {i}/{num_cycles}")
                except Exception:
                    pass

            cycle_start = time.monotonic()
            try:
                t0 = time.monotonic()
                self._scope._camera_driver.stop_grabbing()
                stop_s = time.monotonic() - t0

                if delay_s > 0:
                    time.sleep(delay_s)

                if vary_settings:
                    # Alternate between two presets -- small enough churn
                    # not to dominate the cycle, large enough that GenICam
                    # node-map writes are real.
                    if i % 2 == 0:
                        self._scope.imaging.set_gain(1.0)
                        self._scope.imaging.set_exposure_time(10.0)
                    else:
                        self._scope.imaging.set_gain(4.0)
                        self._scope.imaging.set_exposure_time(50.0)

                t1 = time.monotonic()
                self._scope._camera_driver.start_grabbing()
                start_s = time.monotonic() - t1
            except Exception as e:
                results['errors'].append(
                    f"Cycle {i}: {type(e).__name__}: {e}")
                # Try to leave the camera grabbing for the next iteration;
                # if it fails, the next stop_grabbing will surface it too.
                continue

            cycle_s = time.monotonic() - cycle_start
            cycle_times.append(cycle_s)
            stop_times.append(stop_s)
            start_times.append(start_s)

            if cycle_s >= slow_threshold_s:
                results['slow_cycle_count'] += 1
                # Cap the per-cycle log to keep the JSON small even on
                # pathological runs (every cycle slow).
                if len(results['slow_cycles']) < 50:
                    results['slow_cycles'].append({
                        'idx': i,
                        'cycle_s': round(cycle_s, 4),
                        'stop_s': round(stop_s, 4),
                        'start_s': round(start_s, 4),
                    })

        results['total_elapsed_s'] = round(time.monotonic() - t_overall_start, 3)

        # Restore caller's gain/exposure so vary_settings doesn't leak state.
        try:
            if vary_settings and original_gain is not None:
                self._scope.imaging.set_gain(float(original_gain))
            if vary_settings and original_exposure is not None:
                self._scope.imaging.set_exposure_time(float(original_exposure))
        except Exception as e:
            results['errors'].append(
                f"Restore settings failed: {type(e).__name__}: {e}")

        def _pct(samples, q):
            if not samples:
                return 0.0
            return round(float(np.percentile(samples, q)), 4)

        results['cycle_p50_s'] = _pct(cycle_times, 50)
        results['cycle_p95_s'] = _pct(cycle_times, 95)
        results['cycle_p99_s'] = _pct(cycle_times, 99)
        results['stop_p50_s']  = _pct(stop_times,  50)
        results['stop_p95_s']  = _pct(stop_times,  95)
        results['stop_p99_s']  = _pct(stop_times,  99)
        results['start_p50_s'] = _pct(start_times, 50)
        results['start_p95_s'] = _pct(start_times, 95)
        results['start_p99_s'] = _pct(start_times, 99)

        # Persist to data/camera_timing/ keyed by model + sdk version + delay
        # so a sweep across delays produces one file per data point.
        try:
            import json
            model = results['camera_model'] or 'unknown_camera'
            sdk = results['pylon_version'] or 'unknown_sdk'
            safe_model = str(model).replace(' ', '_').replace('/', '_')
            safe_sdk = str(sdk).replace(' ', '_').replace('/', '_')
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            timing_dir = pathlib.Path(os.path.dirname(__file__)).parent / 'data' / 'camera_timing'
            timing_dir.mkdir(parents=True, exist_ok=True)
            out_path = timing_dir / (
                f'grab_lifecycle_benchmark_{safe_model}_sdk{safe_sdk}_'
                f'delay{int(inter_cycle_delay_ms)}ms_{ts}.json'
            )
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            results['written_to'] = str(out_path)
        except Exception as e:
            results['errors'].append(
                f"Persist failed: {type(e).__name__}: {e}")

        logger.info(
            f"[SCOPE API ] run_grab_lifecycle_benchmark: {num_cycles} cycles, "
            f"delay={inter_cycle_delay_ms}ms, vary={vary_settings} -> "
            f"cycle p50={results['cycle_p50_s']}s p95={results['cycle_p95_s']}s "
            f"p99={results['cycle_p99_s']}s, slow={results['slow_cycle_count']} "
            f"(>={slow_threshold_s}s), total={results['total_elapsed_s']}s"
        )
        return results

    def run_pylon_diagnostic_probe(
        self,
        duration_s: float = 3.0,
        *,
        drain_camera_side_errors: bool = True,
        progress_cb=None,
    ) -> dict:
        """One-shot Pylon-camera diagnostic probe with JSON output.

        Captures camera identity, current configuration, and stream-
        grabber statistics counter deltas over a sampling window of
        ``duration_s`` seconds. Adds host metadata (OS, hostname,
        pypylon + pylon SDK versions) and writes a single JSON file
        to ``data/pylon_probe/`` keyed on
        ``<model>__sn<serial>__fw<firmware>__<host>__dltl<config>__<datetime>.json``.

        Designed for cross-host / cross-camera / cross-firmware
        comparison: the filename pattern keeps a sweep's outputs
        sortable, and ``firmware_version`` + ``dltl_config`` are also
        promoted to top-level JSON keys for filter-by-load.

        Does NOT change grab state. If the camera is not currently
        grabbing, the deltas will be near-zero (stats counters do not
        advance without an active grab loop). Caller is expected to
        be in live preview when calling this method.

        Args:
            duration_s: Sampling window in seconds. Default 3.0
                matches the bench probe shape used to characterize
                dart vs ace 2 on Mac (Firmware DAILY_LOG.md).
            drain_camera_side_errors: When True, drain the camera's
                ``BslErrorPresent`` queue and capture the list of
                opaque error codes (per Basler "evaluated by support",
                no public translation table for ace 2 / dart R).
            progress_cb: Optional ``callback(percent_int, message_str)``
                called at probe start, mid-sample, and end.

        Returns:
            dict: Snapshot from driver plus host / timestamps /
                output_path metadata. Returns
                ``{'connected': False, 'errors': [...]}`` if no
                camera is active. Returns the driver's
                ``{'supported': False, 'reason': ...}`` shape for
                IDS or other non-Pylon drivers.
        """
        if progress_cb is not None:
            try:
                progress_cb(0, 'starting Pylon diagnostic probe')
            except Exception:
                pass

        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            return {'connected': False, 'errors': ['Camera not active']}

        if not hasattr(self._scope._camera_driver, 'read_diagnostic_snapshot'):
            return {
                'connected': False,
                'supported': False,
                'errors': [
                    f'{type(self._scope._camera_driver).__name__} does not implement '
                    f'read_diagnostic_snapshot'
                ],
            }

        # Driver-level snapshot
        snapshot = self._scope._camera_driver.read_diagnostic_snapshot(
            duration_s=duration_s,
            drain_camera_side_errors=drain_camera_side_errors,
        )

        # Non-Pylon stub returns supported=False; pass through unchanged
        if snapshot.get('supported') is False:
            if progress_cb is not None:
                try:
                    progress_cb(100, 'driver does not support diagnostic probe')
                except Exception:
                    pass
            return snapshot

        if progress_cb is not None:
            try:
                progress_cb(70, 'snapshot captured; collecting host metadata')
            except Exception:
                pass

        # Host metadata
        import socket
        import platform as _platform
        host_versions = self._safe_pylon_versions()
        snapshot['host'] = {
            'os': self._human_os_version(),
            'hostname': socket.gethostname(),
            'machine': _platform.machine(),
            'pypylon_version': host_versions['pypylon_version'],
            'pylon_sdk_version': host_versions['pylon_sdk_version'],
        }

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        end_iso = now_utc.isoformat()
        start_iso = (now_utc - datetime.timedelta(
            seconds=snapshot.get('duration_s_actual', duration_s)
        )).isoformat()
        snapshot['timestamps'] = {'start_iso': start_iso, 'end_iso': end_iso}

        # Filter-by-load top-level keys (per v4 author request: easier
        # to grep across many files than parsing camera.firmware_version
        # nested)
        snapshot['firmware_version'] = (
            snapshot.get('camera', {}).get('firmware_version')
        )

        dltl_token = self._dltl_filename_token(snapshot.get('config', {}))
        snapshot['dltl_config'] = dltl_token

        # JSON file write
        try:
            import json
            out_dir = (
                pathlib.Path(os.path.dirname(__file__)).parent
                / 'data' / 'pylon_probe'
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            def _safe_token(v: str | None, fallback: str) -> str:
                s = str(v) if v is not None else fallback
                # Filenames: replace separators that would break the
                # __ split-pattern, and any path separators.
                for bad in (' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|'):
                    s = s.replace(bad, '_')
                return s

            model_t = _safe_token(
                snapshot.get('camera', {}).get('model_name'), 'unknown_model')
            serial_t = _safe_token(
                snapshot.get('camera', {}).get('serial'), 'unknown_serial')
            fw_t = _safe_token(snapshot.get('firmware_version'), 'unknown_fw')
            host_t = _safe_token(
                snapshot['host']['hostname'], 'unknown_host'
            ).replace('.', '_')
            ts_t = now_utc.strftime('%Y%m%dT%H%M%SZ')

            fname = f'{model_t}__sn{serial_t}__fw{fw_t}__{host_t}__{dltl_token}__{ts_t}.json'
            out_path = out_dir / fname
            with open(out_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            snapshot['output_path'] = str(out_path)
        except Exception as e:
            snapshot.setdefault('errors', []).append(
                f'JSON write failed: {type(e).__name__}: {e}'
            )

        if progress_cb is not None:
            try:
                progress_cb(100, 'complete')
            except Exception:
                pass

        return snapshot

    # --- Serial probes ---
    def send_diagnostic_command(
        self,
        target: str,
        command: str,
        *,
        response_numlines: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a single firmware diagnostic command and return the response.

        Wraps the driver's ``exchange_command`` with API-layer logging
        (Rule 13). Diagnostic clients (tech-support report, bench tools)
        MUST go through this method instead of reaching the driver directly
        (LV-24 / LV-32 / LV-40).

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string (e.g. ``'INFO'``, ``'FACTORY'``).
            response_numlines: Forwarded to driver; how many response lines
                to read before returning (driver-specific default if None).
            timeout: Per-call serial timeout in seconds, or None for the
                driver's default.

        Returns:
            str: Response from the board, ``'Board not connected'`` if the
                target board is None/inactive, or ``'Error: <msg>'`` if the
                exchange raised.
        """
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command: {e}')
            return f'Error: {e}'

        if board is None or not getattr(board, 'found', False):
            return 'Board not connected'

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command(target={target}, command={command!r}, '
            f'response_numlines={response_numlines}, timeout={timeout})'
        )
        try:
            kwargs = {}
            if response_numlines is not None:
                kwargs['response_numlines'] = response_numlines
            if timeout is not None:
                kwargs['timeout'] = timeout
            resp = board.exchange_command(command, **kwargs)
            return resp if resp is not None else 'None'
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command({target}, {command!r}) failed: {e}'
            )
            return f'Error: {e}'

    def send_diagnostic_command_multiline(
        self,
        target: str,
        command: str,
        *,
        timeout: float = 60,
        end_markers: list[str] | None = None,
    ) -> 'str | list[str]':
        """Send a firmware diagnostic command expected to return multiple lines.

        For SELFTEST, INFO with multi-line output, etc. Wraps the driver's
        ``exchange_multiline`` with API-layer logging.

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string.
            timeout: Total timeout in seconds.
            end_markers: Substrings marking end-of-response. Default
                ``['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']``.

        Returns:
            Response (driver-defined; typically str or list[str]),
            ``'Board not connected'``, or ``'Error: <msg>'``.
        """
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command_multiline: {e}')
            return f'Error: {e}'

        if board is None or not getattr(board, 'found', False):
            return 'Board not connected'

        if end_markers is None:
            end_markers = ['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command_multiline(target={target}, '
            f'command={command!r}, timeout={timeout}, end_markers={end_markers})'
        )
        try:
            result = board.exchange_multiline(
                command, timeout=timeout, end_markers=end_markers)
            return result if result else 'No response'
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command_multiline({target}, {command!r}) failed: {e}'
            )
            return f'Error: {e}'

    # --- Motor power / driver / fan diagnostics ---
    # Each returns parsed values or None when the firmware does not
    # support the command (legacy 2024-09-10 firmware did not include
    # VOLTAGE / DRVSTAT_<axis> / FANSPEED / FAN). Per Eric: the driver
    # owns firmware-version gating; callers (TSR, future REST
    # diagnostic endpoint) read None as "INCONCLUSIVE -- firmware
    # does not support this probe."

    def read_motor_voltages(self):
        """Read motor-board power rail tolerance diagnostic.

        Returns a dict mapping rail label to volts (or None per rail
        if unparseable), or None when the firmware does not implement
        the VOLTAGE command. See MotorBoard.read_voltages.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_voltages'):
            return None
        return drv.read_voltages()

    def read_motor_drv_status(self, axis: str):
        """Read TMC5072 DRV_STATUS register for an axis.

        Returns the raw register value as int (caller decodes bits),
        or None when the firmware does not implement DRVSTAT_<axis>.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_drv_status'):
            return None
        return drv.read_drv_status(axis)

    def read_motor_fanspeed(self):
        """Read motor-board fan tachometer RPM.

        Returns RPM as int (0 if no tach wire) or None when firmware
        does not implement FANSPEED.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_fanspeed'):
            return None
        return drv.read_fanspeed()

    def set_motor_fan_duty(self, duty_pct: int) -> bool:
        """Set motor-board fan PWM duty cycle (0..100).

        Returns True if firmware accepted the command, False if firmware
        does not implement FAN:<duty> or no motor driver is present.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'set_fan_duty'):
            return False
        return drv.set_fan_duty(duty_pct)

    # --- LED engineering mode (LEDREADS / SELFTEST handshake) ---
    # Open-coded FACTORY / Y / Q sequences in callers were leaving the
    # LED board wedged when LEDREADS or SELFTEST timed out mid-eng-mode
    # -- Q was sent without the post-Q drain + end-marker check that the
    # driver-canonical methods do, so the firmware was left in a state
    # where every subsequent LED command returned ''. These sub-API
    # entries keep the careful handshake as the single canonical
    # implementation.

    def enter_led_engineering_mode(self, timeout: float = 5.0) -> bool:
        """Enter LED engineering mode via the driver-canonical handshake.

        Returns True on success, False when the LED driver is absent or
        does not expose engineering-mode entry (legacy LED firmware
        predating the FACTORY/Y/Q protocol).
        """
        drv = getattr(self._scope, '_led_driver', None)
        if drv is None or not hasattr(drv, 'enter_engineering_mode'):
            return False
        try:
            return drv.enter_engineering_mode(timeout=timeout)
        except Exception:
            return False

    def exit_led_engineering_mode(self):
        """Exit LED engineering mode via the driver-canonical handshake.

        Driver method drains and sleeps after Q so the LED firmware
        actually transitions out of eng mode.
        """
        drv = getattr(self._scope, '_led_driver', None)
        if drv is None or not hasattr(drv, 'exit_engineering_mode'):
            return None
        try:
            return drv.exit_engineering_mode()
        except Exception:
            return None

    # --- Phase 5 helper forwarders (5b-introduced; bodies co-relocate 5c) ---
    # These four helpers are exclusive to diagnostic methods. 5b adds
    # forwarders so tests can target DiagnosticsAPI._X(...) directly;
    # 5c moves the bodies here and retires the Lumascope copies. Lazy
    # imports avoid the diagnostics.py <-> _lumascope.py cycle.

    @staticmethod
    def _safe_pylon_versions() -> dict:
        """Best-effort capture of pypylon + pylon SDK runtime versions.

        Both reads are wrapped: pypylon may not be installed (FX2-only
        installs), or the runtime version helper may have been renamed
        between SDK versions.
        """
        out = {'pypylon_version': None, 'pylon_sdk_version': None}
        try:
            import pypylon as _pyp
            out['pypylon_version'] = getattr(_pyp, '__version__', None)
        except Exception:
            pass
        try:
            from pypylon import pylon as _pylon
            for fn_name in ('GetPylonVersion', 'GetVersionString'):
                fn = getattr(_pylon, fn_name, None)
                if callable(fn):
                    try:
                        out['pylon_sdk_version'] = str(fn())
                        break
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    @staticmethod
    def _human_os_version() -> str:
        """Render OS version in a form humans can recognise.

        ``platform.release()`` on macOS returns the Darwin kernel
        version (e.g. ``24.6.0``) which nobody can map to "macOS 14.x"
        by inspection. ``platform.mac_ver()[0]`` returns the actual
        macOS version (e.g. ``14.5``); equivalent on Windows is
        ``platform.win32_ver()[0]``. Falls back to system + release
        on Linux / unknown.
        """
        import platform as _pl
        sys_name = _pl.system()
        try:
            if sys_name == 'Darwin':
                mac = _pl.mac_ver()[0]
                if mac:
                    return f'macOS {mac}'
            elif sys_name == 'Windows':
                win = _pl.win32_ver()[0]
                if win:
                    return f'Windows {win}'
        except Exception:
            pass
        return f'{sys_name} {_pl.release()}'

    @staticmethod
    def _dltl_filename_token(config: dict) -> str:
        """Encode the DLTL config as a short filename-safe token.

        Examples:
            DLTL Off                  -> 'dltloff'
            DLTL On at 160 MB/s       -> 'dltl160M'
            DLTL On at 197.43 MB/s    -> 'dltl197M' (rounded)
            anything else / missing   -> 'dltlunknown'

        ``int(round(...))`` handles non-round sweep values cleanly --
        v4 author flagged the case where a sweep set DLTL to an
        intermediate value with sub-MB/s precision.
        """
        mode = config.get('dltl_mode')
        if isinstance(mode, str) and mode.lower() == 'off':
            return 'dltloff'
        value = config.get('dltl_value_bps')
        if isinstance(value, (int, float)) and value > 0:
            return f'dltl{int(round(value / 1_000_000))}M'
        return 'dltlunknown'

    def _diagnostic_target_board(self, target: str):
        """Resolve a diagnostic-target string ('led' | 'motor') to a driver board.

        Internal helper for ``send_diagnostic_command*``. Raises
        ``ValueError`` for an unknown target so a typo in tech-support
        code fails loudly rather than silently picking the wrong board.
        """
        target = target.lower() if isinstance(target, str) else target
        if target == 'led':
            return self._scope._led_driver
        if target in ('motor', 'motion'):
            return self._scope._motion_driver
        raise ValueError(
            f"send_diagnostic_command: unknown target {target!r} "
            f"(expected 'led' or 'motor')")
