#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib
import threading
import time
from lvp_logger import logger

from drivers.serialboard import SerialBoard, ProtocolVersion
from drivers.registry import motor_registry
from modules.exceptions import HardwareError
from modules.motorconfig import MotorConfig


@motor_registry.register('rp2040', priority=100)
class MotorBoard(SerialBoard):

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, motorconfig_defaults_file: pathlib.Path | None = None, **kwargs):
        self._state_lock = threading.Lock()
        self.overshoot = False
        self._has_turret = False
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self._fullinfo = None
        self._connect_fails = 0
        self._connect_log_suppressed = False
        # FW4.0 motion event subscribers — None until set_arrived_callback /
        # set_homed_callback is called. The dispatcher installs itself on
        # SerialBoard.on_event when either subscriber becomes non-None.
        self._arrived_callback = None
        self._homed_callback = None

        # Load hardware config (per-unit values from motorconfig.json, with defaults fallback)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path("data/motorconfig_defaults.json")
        self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)

        # Default timeout 5s for regular commands. Long-running commands
        # (HOME, CALIBRATE) pass explicit timeout overrides (H15).
        super().__init__(vid=0x2E8A, pid=0x0005, label='[XYZ Class ]',
                         timeout=5, write_timeout=5)

        # Backward-compatible alias for lock name
        self.thread_lock = self._lock

        # 1. Build cached values from defaults
        self._rebuild_cached_values()
        # 2. Open port, reset firmware, verify connection
        self._initial_connect()
        # 3. Load per-unit config from board, rebuild cache with real values
        self._load_board_config()

    # ------------------------------------------------------------------
    # Dual-protocol dispatch (FW4.0 V4 + LEGACY v3.0.x).
    # Both lanes are permanent per primary-session posture (2026-04-21).
    # Callers never need to know which is active — legacy methods gain
    # V4 branches internally, gated by _use_v4() which capability-probes
    # via has_feature(). No "deprecated" framing; LEGACY stays equal-class
    # for the life of the LVP 4.x series.
    # ------------------------------------------------------------------
    def _use_v4(self):
        """True iff the connected board speaks FW4.0 AND advertises
        positions (the baseline motor capability). Defensive getattr for
        tests that construct MotorBoard via __new__."""
        if getattr(self, 'protocol_version', None) != ProtocolVersion.V4:
            return False
        # 'positions' is the baseline — every FW4.0 motor firmware has it
        # (see docs/FW40_COMMAND_REFERENCE.md §4). Using it as the gate
        # means a partial-feature FW4.0 build still routes correctly.
        return 'positions' in getattr(self, 'features', [])

    # ------------------------------------------------------------------
    # Motion event subsystem (FW4.0 EVENTS ON).
    #
    # The LVP motion-monitor previously polled STATUS at 50 Hz holding
    # SerialBoard._lock — ~32 ms lock-held per call, observed as 1.80 s
    # of contention on a 36-step bench run (2026-04-13 profile). FW4.0
    # firmware emits {"event":"arrived","axis":X,"pos":N} on rising
    # edge of (pos_reached AND vel_zero) at 50 Hz; subscribing here lets
    # the API-layer motion monitor drop its poll to a 2 Hz watchdog.
    #
    # Callers install an arrived-event callback via set_arrived_callback;
    # a homed-event callback via set_homed_callback. MotorBoard filters
    # event messages from the SerialBoard on_event stream by event name
    # and routes. Multiple subscribers are not supported (one callback
    # each) — if that becomes a need, expand to a list.
    # ------------------------------------------------------------------
    def _on_event_dispatch(self, msg):
        """Router installed on SerialBoard.on_event; filters events by
        name and calls the appropriate subscriber. Runs on the
        exchange_json read thread (same thread as the command caller)
        so it must be quick — subscribers should enqueue to the API
        layer's own thread, not do work inline here."""
        evt = msg.get('event')
        if evt == 'arrived':
            cb = self._arrived_callback
            if cb is not None:
                try:
                    cb(msg.get('axis'), msg.get('pos'))
                except Exception as e:
                    logger.error(f'[XYZ Class ] arrived callback raised: {e}')
        elif evt == 'homed':
            cb = self._homed_callback
            if cb is not None:
                try:
                    cb(msg.get('axis'), msg.get('pos'))
                except Exception as e:
                    logger.error(f'[XYZ Class ] homed callback raised: {e}')

    def set_arrived_callback(self, fn):
        """Install a callback for 'arrived' events. fn(axis_str, pos_int).
        Pass None to clear. Wiring on_event is idempotent — subsequent
        calls just swap the subscriber, not re-install the dispatcher."""
        self._arrived_callback = fn
        self._install_event_dispatcher()

    def set_homed_callback(self, fn):
        """Install a callback for 'homed' events. fn(axis_str, pos_int)."""
        self._homed_callback = fn
        self._install_event_dispatcher()

    def _install_event_dispatcher(self):
        """Wire _on_event_dispatch onto the SerialBoard.on_event slot if
        not already done. Idempotent."""
        if self.on_event is not self._on_event_dispatch:
            self.on_event = self._on_event_dispatch

    def motion_events_on(self):
        """Enable the firmware's 50 Hz arrived/homed push subsystem.
        Returns True on success (V4 board acknowledged), False on LEGACY
        (subsystem doesn't exist) or V4 communication failure.

        Caller must have set an arrived/homed callback first (otherwise
        events are emitted and dropped silently)."""
        if not self._use_v4() or not self.has_feature('events'):
            return False
        self._install_event_dispatcher()
        resp = self.exchange_json({'cmd': 'EVENTS', 'mode': 'ON'})
        return resp is not None and resp.get('ok') is True

    def motion_events_off(self):
        """Disable firmware push events. The callback slot stays
        installed — re-enabling works without re-registering."""
        if not self._use_v4() or not self.has_feature('events'):
            return False
        resp = self.exchange_json({'cmd': 'EVENTS', 'mode': 'OFF'})
        return resp is not None and resp.get('ok') is True

    def positions_batch(self):
        """Batch read of all present axis positions in a single round-trip.
        FW4.0 only — replaces 4x current_pos_steps for the motion-monitor
        path. Returns {'X': int, 'Y': int, 'Z': int, ...} or None on
        LEGACY / failure. Omits axes that are not present."""
        if not self._use_v4() or not self.has_feature('positions'):
            return None
        resp = self.exchange_json({'cmd': 'POSITIONS'})
        if resp is None or resp.get('ok') is not True:
            return None
        out = {}
        for ax in ('X', 'Y', 'Z', 'T'):
            if ax in resp:
                out[ax] = resp[ax]
        return out

    def _v4_home_wait(self, payload, total_timeout):
        """Issue an async FW4.0 HOME command and block until STATUS
        reports homing:false, preserving the sync True/False contract
        of the legacy home methods. Returns (ok, result_str) where
        ok=True on clean completion, False on firmware error/timeout.

        Polls STATUS at 2 Hz. API-layer consumers subscribed to the
        homed event callback get per-axis updates faster than this
        poll; this wait is only for the driver method's sync return."""
        resp = self.exchange_json(payload, timeout=5)
        if resp is None:
            return False, 'no response'
        if resp.get('ok') is not True:
            return False, resp.get('msg', str(resp))
        deadline = time.monotonic() + total_timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            status = self.exchange_json({'cmd': 'STATUS'}, timeout=2)
            if status is None:
                continue
            if not status.get('homing', False):
                return True, status.get('result', 'OK')
        return False, f'timed out after {total_timeout}s'

    def _rebuild_cached_values(self):
        """Recompute cached values from motorconfig.

        Called at init (with defaults only) and again after
        update_from_board() merges per-unit board data inside connect().

        NOTE: This must NOT call connect(). connect() calls this method
        after update_from_board(), so calling connect() here would recurse
        and attempt to reopen the serial port while it's already open —
        causing PermissionError on Windows. (#610)
        """
        self.backlash = self.motorconfig.antibacklash_um('Z')
        self.axes_config = {
            'Z': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('Z'),
                },
                'move_func': self.z_um2ustep
            },
            'X': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('X'),
                },
                'move_func': self.xy_um2ustep
            },
            'Y': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('Y'),
                },
                'move_func': self.xy_um2ustep
            },
            'T': {
                'move_func': self.t_pos2ustep
            }
        }

    def _initial_connect(self):
        """Called once from __init__ to establish the first connection."""
        logger.info('[XYZ Class ] _initial_connect() — first connection attempt')
        try:
            self.connect()
        except Exception:
            logger.error('[XYZ Class ] _initial_connect() failed')
            raise

    def _load_board_config(self):
        """Read per-unit config from connected board and merge into motorconfig.

        Called once after connect() succeeds. Separate from connect() because
        connect's job is opening the port — config loading is a post-connect step.
        """
        try:
            board_cfg = self.get_config()
            if board_cfg:
                self.motorconfig.update_from_board(board_cfg)
                self._rebuild_cached_values()
                logger.info(f'[XYZ Class ] Board config merged: model={self.motorconfig.model()}, '
                            f'SN={self.motorconfig.serial_number()}, '
                            f'Z_usteps/mm={self.motorconfig.usteps_per_mm("Z")}')
        except Exception as e:
            logger.warning(f'[XYZ Class ] Board config load failed (using defaults): {e}')

    def _on_disconnect(self):
        """Clear cached firmware info on disconnect (called under self._lock)."""
        with self._state_lock:
            self._fullinfo = None
            self.initial_homing_complete = False
            self.initial_t_homing_complete = False
        self._accel_cache = None
        logger.info('[XYZ Class ] Motor state cache cleared on disconnect')

    def _connect_bench_callables(self):
        """Driver methods benched at connect-time (release gate §2.3).

        `fullinfo` is the one read-path that exists on both v3.0.x
        (FULLINFO, drain-sleep penalty) and FW4.0 (INFO, no drain),
        so its round-trip latency is the core cross-firmware
        comparison point.
        """
        return [('fullinfo', self.fullinfo)]

    def stop(self):
        """Emergency-halt all motors — aborts async HOME and clamps targets.

        FW4.0 (V4): STOP command. Firmware aborts any running async op
        (HOME) via fw.async_set_abort(), then writes actual→target for
        every present axis via the TMC5072 ramp controller (immediate
        stop at current position).

        LEGACY (v3.0.x): STOP command. Writes actual→target for all 4
        axes via SPI (`motorstop()` in v3.0.x firmware). No async-op
        abort path — v3.0.x homing was synchronous.

        Returns a normalized dict so callers don't branch on protocol:
            {'ok': bool, 'stopped': bool,
             'positions': {axis: pos, ...} | None,
             'response': raw_response_str | None}

        `positions` is None on LEGACY (v3.0.x returns `'STOPPED'` with
        no axis detail). Returns None on driver error.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'STOP'}, timeout=5)
            if resp is None:
                return None
            return {
                'ok': bool(resp.get('ok')),
                'stopped': bool(resp.get('stopped')),
                'positions': {
                    ax: resp[ax] for ax in ('X', 'Y', 'Z', 'T')
                    if ax in resp and isinstance(resp.get(ax), (int, float))
                } or resp.get('positions'),
                'response': None,
            }

        resp = self.exchange_command('STOP', timeout=5)
        if resp is None:
            return None
        return {
            'ok': True,
            'stopped': 'STOPPED' in str(resp).upper(),
            'positions': None,
            'response': str(resp).strip(),
        }

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        # Note: _lock is an RLock (from SerialBoard), so re-entrant acquisition
        # by _open_serial, _reset_firmware, exchange_command etc. is safe.
        with self._lock:
            try:
                # Skip if already connected
                if self.driver is not None and self.driver.is_open:
                    logger.debug(f'[XYZ Class ] connect() skipped — already connected on {self.port}')
                    return

                logger.info(f'[XYZ Class ] connect() starting on {self.port}')
                self._open_serial()
                logger.info(f'[XYZ Class ] connect() port opened: {self.port}')

                # Legacy port reset: close and reopen to flush USB CDC
                # buffers on Windows. Has existed since original code.
                self.driver.close()
                logger.debug(f'[XYZ Class ] connect() port closed for reset')
                time.sleep(0.05)  # brief pause for Windows to release port
                self.driver.open()
                logger.debug(f'[XYZ Class ] connect() port reopened after reset')

                self._connect_fails = 0
                self._connect_log_suppressed = False

                self._reset_firmware()
                info = self.fullinfo()
                with self._state_lock:
                    self._fullinfo = info

                logger.info('[XYZ Class ] Connected to motor controller')
                # Fire the connect-time latency fingerprint — SerialBoard.connect
                # has this call too, but MotorBoard.connect is an override that
                # does not delegate up, so the hook has to fire here as well.
                # Caught on bench 2026-04-24: SN 115 LED had a populated
                # connect_latency_summary but motor stayed None.
                self._run_connect_latency_bench()
            except Exception as e:
                self._close_driver()
                self._connect_fails += 1
                if self._connect_fails >= 10 and not self._connect_log_suppressed:
                    logger.critical('[XYZ Class ] MotorBoard.connect() failed 10 times — suppressing further connect errors (other logging continues)')
                    self._connect_log_suppressed = True
                if not self._connect_log_suppressed:
                    logger.error(f'[XYZ Class ] MotorBoard.connect() failed: {e}')


    # Firmware 1-14-2023 commands include
    # 'QUIT'
    # 'INFO'
    # 'HOME'
    # 'ZHOME'
    # 'THOME'
    # 'ACTUAL_R'
    # 'ACTUAL_W'
    # 'TARGET_R'
    # 'TARGET_W'
    # 'STATUS_R'
    # 'SPI'

    #----------------------------------------------------------
    # Informational Functions
    #----------------------------------------------------------
    def fullinfo(self):
        if self._use_v4():
            # FW4.0 merged FULLINFO into INFO per the spec §5.
            resp = self.exchange_json({'cmd': 'INFO'})
            if resp is None or resp.get('ok') is not True:
                logger.error('[XYZ Class ] INFO V4 returned None/error — board disconnected?')
                return {"model": "unknown", "serial_number": "unknown"}
            model = resp.get('model', 'unknown')
            if isinstance(model, str) and model and model[-1] == 'T':
                with self._state_lock:
                    self._has_turret = True
            serial_number = resp.get('serial', 'unknown')
            return {
                "model": model,
                "serial_number": serial_number,
                "_raw": resp,
                "_info": resp,  # structured dict for V4 consumers
            }

        info = self.exchange_command("FULLINFO")
        logger.info('[XYZ Class ] MotorBoard.fullinfo(): %s', info, extra={'force_error': True})
        if info is None:
            logger.error('[XYZ Class ] FULLINFO returned None — board disconnected?')
            return {"model": "unknown", "serial_number": "unknown"}
        try:
            parts = info.split()
            model = parts[parts.index("Model:") + 1]
            if model[-1] == "T":
                with self._state_lock:
                    self._has_turret = True
            serial_number = parts[parts.index("Serial:") + 1]
        except (ValueError, IndexError) as e:
            logger.error(f'[XYZ Class ] Failed to parse FULLINFO response: {info!r} ({e})')
            return {"model": "unknown", "serial_number": "unknown"}
        return {
            "model": model,
            "serial_number": serial_number,
            "_raw": info,  # Cached raw response for detect_present_axes()
        }


    def get_microscope_model(self):
        with self._state_lock:
            info = self._fullinfo
        if info is None:
            # Connection never completed (port held / open() failed) so
            # FULLINFO was never cached. Defensive: the registry's
            # is_connected() gate (commit a5f5eff) should keep callers
            # from ever seeing a real MotorBoard with _fullinfo=None,
            # but defense-in-depth per Rule 8 — driver methods must
            # never raise on a disconnected instance.
            return None
        return info.get('model')

    def detect_present_axes(self):
        """Detect which axes are present on this board.

        Uses cached FULLINFO from connect() if available, avoiding
        an unnecessary serial round-trip.
        Returns list of axis letters, e.g. ['X', 'Y', 'Z', 'T'] or ['Z', 'T'].
        """
        # Use cached fullinfo if available (set during connect)
        with self._state_lock:
            info = self._fullinfo

        # V4 fast path: structured INFO.axes list.
        if info is not None and isinstance(info.get('_info'), dict):
            axes_list = info['_info'].get('axes')
            if isinstance(axes_list, list):
                return [ax for ax in axes_list if ax in ('X', 'Y', 'Z', 'T')]

        # V4 live query if no cache.
        if info is None and self._use_v4():
            resp = self.exchange_json({'cmd': 'INFO'})
            if resp is not None and isinstance(resp.get('axes'), list):
                return [ax for ax in resp['axes'] if ax in ('X', 'Y', 'Z', 'T')]

        if info is not None:
            resp = info.get('_raw', '')
        else:
            resp = self.exchange_command('FULLINFO') or ''
        axes = []
        for axis in ('X', 'Y', 'Z', 'T'):
            if f'{axis} present: True' in resp or f'{axis} present:True' in resp:
                axes.append(axis)
        return axes

    def current_pos_steps(self, axis):
        """Get current position in raw microsteps (no unit conversion).

        Returns int or None on failure.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'POS_READ', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                return None
            return resp.get('position')
        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] current_pos_steps({axis}) failed: {e}')
            return None

    def target_pos_steps(self, axis):
        """Get target position in raw microsteps (no unit conversion).

        Returns int or None on failure.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'POS_READ', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                return None
            # FW4.0 POS_READ returns both position (actual) and target.
            return resp.get('target')
        try:
            response = self.exchange_command('TARGET_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] target_pos_steps({axis}) failed: {e}')
            return None

    #----------------------------------------------------------
    # Acceleration control functions
    #----------------------------------------------------------

    # Cache for acceleration limits — read once from firmware, reuse thereafter.
    # Invalidated on reconnect via _on_disconnect().
    _accel_cache: dict = None

    # Get single acceleration limit for a specific axis and parameter
    def acceleration_limit(self, axis: str, parameter: str) -> int:
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return 0

        # Return cached value if available
        cache_key = f"{axis}_{parameter}"
        if self._accel_cache is not None and cache_key in self._accel_cache:
            return self._accel_cache[cache_key]

        DEFAULT_ACCELERATION_LIMIT = 30000

        if self._use_v4():
            # FW4.0 routes through MOTOR_PARAM (structural, TMC-agnostic)
            # instead of raw SPI_REG. Param name matches spec §4.
            param_map = {'acceleration': 'AMAX', 'deceleration': 'DMAX'}
            param = param_map[parameter]
            resp = self.exchange_json({'cmd': 'MOTOR_PARAM', 'axis': axis,
                                        'param': param})
            if resp is None or resp.get('ok') is not True:
                value = DEFAULT_ACCELERATION_LIMIT
                logger.debug(f'[XYZ Class ] acceleration_limit({axis},{parameter}) V4 failed, using default {DEFAULT_ACCELERATION_LIMIT}: {resp}')
            else:
                value = int(resp.get('value', DEFAULT_ACCELERATION_LIMIT))
                logger.info(f'[XYZ Class ] MotorBoard.acceleration_limit({axis},{parameter}) V4: {value}')
            if self._accel_cache is None:
                self._accel_cache = {}
            self._accel_cache[cache_key] = value
            return value

        parameter_map = {
            'acceleration': 'A',
            'deceleration': 'D'
        }

        parameter_char = parameter_map[parameter]
        command = f"{parameter_char}MAX{axis}"
        using_default = False
        try:
            resp = self.exchange_command(command)

            # In case firmware doesn't support retrieving the acceleration limits
            if resp is None or resp.startswith("ERROR"):
                raise ValueError(f"Firmware returned ERROR for {command}")

            # Extra protection for now in case motorboard responds with a different string that doesnt start with ERROR
            if not resp.isdigit():
                raise ValueError(f"Non-numeric response for {command}: {resp}")

        except Exception:
            resp = DEFAULT_ACCELERATION_LIMIT
            using_default = True

        if using_default:
            logger.debug(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): firmware does not support, using default {DEFAULT_ACCELERATION_LIMIT}')
        else:
            logger.info(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): {resp}')

        value = int(resp)

        # Cache the result
        if self._accel_cache is None:
            self._accel_cache = {}
        self._accel_cache[cache_key] = value

        return value


    def _acceleration_validate_inputs(self, axis: str, parameter: str):
        config = self._acceleration_supported_info()
        if axis not in config['axes']:
            raise NotImplementedError(f"Support for acceleration limit on axis {axis} not implemented")

        if parameter not in config['parameters']:
            raise NotImplementedError(f"Support for acceleration limit parameter {parameter} not implemented.")

        return True


    def _acceleration_supported_info(self):
        return {
            'axes': ('X','Y'),
            'parameters': ('acceleration', 'deceleration')
        }

    # Get all acceleration limits for all axes and parameters
    def acceleration_limits(self) -> dict[str, dict[str, int]]:
        limits = {}
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            limits[axis] = {}
            for parameter in config['parameters']:
                limits[axis][parameter] = self.acceleration_limit(axis=axis, parameter=parameter)

        return limits


    # Sets the percentage acceleration/deceleration limit (of max) for a single axis/parameter
    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int):
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return

        if (val_pct < 1) or (val_pct > 100):
            raise ValueError(f"Acceleration limit of {val_pct}% is out of bounds. Must be between 1 and 100.")

        limit = self.acceleration_limit(axis=axis, parameter=parameter)
        setpoint = round(limit*(val_pct/100))

        if self._use_v4():
            param_map = {'acceleration': 'AMAX', 'deceleration': 'DMAX'}
            resp = self.exchange_json({'cmd': 'MOTOR_PARAM', 'axis': axis,
                                        'param': param_map[parameter],
                                        'value': int(setpoint)})
            if resp is None or resp.get('ok') is not True:
                logger.warning(f'[XYZ Class ] set_acceleration_limit({axis},{parameter},{val_pct}%) V4 failed: {resp}')
                return
            logger.info(f'[XYZ Class ] MotorBoard.set_acceleration_limit({axis}, {parameter}, {val_pct}%) V4')
            return

        SPI_ADDRS = {
            'X': {
                'acceleration': 0x26,
                'deceleration': 0x28,
            },
            'Y': {
                'acceleration': 0x46,
                'deceleration': 0x48,
            },
        }

        self.spi_write(
            axis=axis,
            addr=SPI_ADDRS[axis][parameter],
            payload=setpoint
        )
        logger.info(f"[XYZ Class ] MotorBoard.set_acceleration_limit({axis}, {parameter}, {val_pct}%)")


    # Sets the percentage acceleration/deceleration (of max) for all supported axes/parameters
    def set_acceleration_limits(self, val_pct):
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            for parameter in config['parameters']:
                self.set_acceleration_limit(axis=axis, parameter=parameter, val_pct=val_pct)

    #----------------------------------------------------------
    # SPI-direct related functions
    #----------------------------------------------------------
    def spi_read(self, axis: str, addr: int) -> str:
        if self._use_v4():
            # FW4.0 SPI_REG: firmware handles the read-dummy-payload detail
            # internally and returns the register value as a hex string.
            resp = self.exchange_json({'cmd': 'SPI_REG', 'axis': axis,
                                        'addr': f'0x{addr:02x}'})
            if resp is None or resp.get('ok') is not True:
                logger.warning(f'[XYZ Class ] spi_read({axis}, 0x{addr:02x}) V4 failed: {resp}')
                return None
            value = resp.get('value')
            logger.debug(f'[XYZ Class ] MotorBoard.spi_read({axis}, 0x{addr:02x}) V4 -> {value}')
            return value

        # Add a dummy payload of "00" to the end in order for the firmware to not error out on a read.
        # It is expecting a payload.
        command = f"SPI{axis}0x{addr:02x}00"
        resp = self.exchange_command(command)
        logger.debug(f"[XYZ Class ] MotorBoard.spi_read({axis}, 0x{addr:02x}): {command} -> {resp}")
        return resp


    def spi_write(self, axis: str, addr: int, payload: int | str) -> str:
        """Write to TMC motor driver SPI register.

        Args:
            axis: Motor axis ('X', 'Y', 'Z', 'T').
            addr: SPI register address (0x00-0x7F; write offset 0x80 added automatically).
            payload: Value to write (decimal integer or string representation).
        """
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f"Invalid axis {axis!r}")
        if not (0 <= addr <= 0x7F):
            raise ValueError(f"SPI address 0x{addr:02X} out of range [0x00-0x7F]")
        if self._use_v4():
            # FW4.0 SPI_REG with an int 'value' signals write; firmware
            # applies the 0x80 write bit internally. Raw addr is passed
            # (no host-side offset addition).
            resp = self.exchange_json({'cmd': 'SPI_REG', 'axis': axis,
                                        'addr': f'0x{addr:02x}',
                                        'value': int(payload)})
            if resp is None or resp.get('ok') is not True:
                logger.warning(f'[XYZ Class ] spi_write({axis}, 0x{addr:02x}, {payload}) V4 failed: {resp}')
                return None
            logger.debug(f'[XYZ Class ] MotorBoard.spi_write({axis}, 0x{addr:02x}, {payload}) V4 -> {resp}')
            return resp.get('value')

        WRITE_OFFSET = 0x80
        write_addr = addr + WRITE_OFFSET
        command = f"SPI{axis}0x{write_addr:02x}{int(payload)}"
        resp = self.exchange_command(command)
        logger.debug(f"[XYZ Class ] MotorBoard.spi_write({axis}, 0x{addr:02x}, {payload}): {command} -> {resp}")
        return resp


    #----------------------------------------------------------
    # Precision mode — controls motor stop accuracy
    #----------------------------------------------------------

    # TMC5072 VSTOP register addresses per axis.
    # VSTOP sets the velocity threshold for declaring "stopped" —
    # lower = more accurate final position, slightly slower settle.
    _VSTOP_ADDR = {
        'X': 0x2B,  # VSTOP_M1 on XY chip
        'Y': 0x4B,  # VSTOP_M2 on XY chip
        'Z': 0x4B,  # VSTOP_M2 on ZT chip
        'T': 0x2B,  # VSTOP_M1 on ZT chip
    }
    _VSTOP_NORMAL = 1000    # factory default — fast but overshoots
    _VSTOP_PRECISION = 100  # accurate stop position

    def set_precision_mode(self, axis: str, enabled: bool):
        """Set motor precision mode for an axis.

        Precision mode uses a lower VSTOP threshold so the motor fully
        decelerates before reporting target reached. Use for autofocus
        fine passes and any measurement that needs accurate positioning.

        Normal mode uses a higher VSTOP for faster moves where overshoot
        is acceptable (coarse AF pass, user jogging, homing approach).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.
        """
        if axis not in self._VSTOP_ADDR:
            logger.warning(f'[XYZ Class ] set_precision_mode: invalid axis {axis}')
            return
        vstop = self._VSTOP_PRECISION if enabled else self._VSTOP_NORMAL
        addr = self._VSTOP_ADDR[axis]
        self.spi_write(axis, addr, str(vstop))
        logger.info(f'[XYZ Class ] {axis} precision_mode={enabled} (VSTOP={vstop})')

    #----------------------------------------------------------
    # Z (Focus) Functions
    # Stock actuator = 0.30 mm pitch.  (1 rev/0.30 mm) x (200 steps/rev) x (256 usteps/step) = 170667 ustep/mm
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        um = (ustep * 1000 / usteps_per_mm)
        return um

    def z_um2ustep(self, um):
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def zhome(self):
        """Home the objective. Returns True on success, False on failure."""
        if self._use_v4():
            ok, result = self._v4_home_wait({'cmd': 'HOME', 'axis': 'Z'},
                                            total_timeout=15)
            logger.info(f'[XYZ Class ] MotorBoard.zhome() V4 -> ok={ok} result={result}')
            if not ok:
                logger.error(f'[XYZ Class ] zhome() V4 failed: {result}')
            return ok

        resp = self.exchange_command('ZHOME', timeout=15)
        logger.info(f'[XYZ Class ] MotorBoard.zhome() -> {resp}')
        if resp is None:
            logger.error('[XYZ Class ] zhome(): no response (timeout or disconnect)')
            return False
        success = 'successful' in resp.lower() or 'complete' in resp.lower()
        if not success:
            logger.error(f'[XYZ Class ] zhome() failed: {resp}')
        return success

    #----------------------------------------------------------
    # XY Stage Functions
    # Stock actuator = 2.54mm pitch.  (1 rev/2.540 mm) x (200 steps/rev) x (256 usteps/step) = 20157 ustep/mm
    #----------------------------------------------------------

    def xy_ustep2um(self, ustep):
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        um = (ustep * 1000 / usteps_per_mm)
        return um

    def xy_um2ustep(self, um):
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def home_axis(self, axis):
        """Home a single axis. Returns True on success, False on failure.

        Dispatcher for per-axis home requests:
          Z → zhome()
          T → thome()
          X or Y → XY group home (FW4.0 'HOME XY' / v3.0.x full HOME).

        Neither protocol supports X-only or Y-only homing — the firmware
        always pairs X+Y because the hardware does too. Callers asking
        for X or Y get the closest available primitive: 'HOME XY' on
        FW4.0 (no Z/T touched) and full HOME on v3.0.x (Z+T+XY, since
        v3.0.x has no XYHOME).
        """
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f'Invalid axis {axis!r}')
        if axis == 'Z':
            return self.zhome()
        if axis == 'T':
            return self.thome()
        # X or Y → XY group
        if self._use_v4():
            ok, _result = self._v4_home_wait({'cmd': 'HOME', 'axis': 'XY'},
                                             total_timeout=30)
            return ok
        # v3.0.x has no XYHOME; fall through to full HOME.
        return self.home()

    def home(self):
        """Send HOME to firmware and home every axis the board has.

        The firmware's xyzhome routine homes Z, then T, then attempts X/Y.
        On a full XYZ(T) board, the response is 'XYZ home complete'. On a
        Z-only board (LS820 bench), Z (and T if present) get homed first
        and the firmware then returns 'ERROR: X not present' — the home
        DID succeed for the axes the board has, so this counts as
        success. Real failures (no response, hardware error, or partial
        home aborted by Z/T error) return False.

        Returns True on full or partial success, False on real failure.
        """
        if self._use_v4():
            ok, result = self._v4_home_wait({'cmd': 'HOME'}, total_timeout=30)
            logger.info(f'[XYZ Class ] MotorBoard.home() V4 -> ok={ok} result={result}',
                        extra={'force_error': True})
            result_str = str(result)
            if ok:
                with self._state_lock:
                    self.initial_homing_complete = True
                return True
            # Partial-home semantics match LEGACY: a "not present" report
            # for X or Y still means the axes the board has were homed
            # successfully.
            if 'not present' in result_str.lower() and ('X' in result_str or 'Y' in result_str):
                logger.info(f'[XYZ Class ] partial home V4 (X/Y not present): {result_str}')
                with self._state_lock:
                    self.initial_homing_complete = True
                return True
            logger.error(f'[XYZ Class ] home() V4 failed: {result_str}')
            return False

        resp = self.exchange_command('HOME', timeout=30)
        logger.info(f'[XYZ Class ] MotorBoard.home() -> {resp}', extra={'force_error': True})
        if resp is None:
            logger.error('[XYZ Class ] home(): no response (timeout or disconnect)')
            return False
        if 'XYZ home complete' in resp:
            with self._state_lock:
                self.initial_homing_complete = True
            return True
        # Partial home: firmware homed Z (and T if present) before
        # reporting that X or Y is not physically wired on this board.
        # The reference position for the present axes is valid.
        if ('not present' in resp) and ('X' in resp or 'Y' in resp):
            logger.info(f'[XYZ Class ] partial home (X/Y not present on this board): {resp}')
            with self._state_lock:
                self.initial_homing_complete = True
            return True
        logger.error(f'[XYZ Class ] home() failed: {resp}')
        return False

    def has_homed(self):
        with self._state_lock:
            return self.initial_homing_complete

    def xycenter(self):
        """ Home the stage which also homes the objective first """
        logger.info('[XYZ Class ] MotorBoard.xycenter()')
        if self._use_v4():
            # HOME FULL per spec §4: home XYZ then move XY to center.
            ok, result = self._v4_home_wait({'cmd': 'HOME', 'mode': 'FULL'},
                                            total_timeout=30)
            if not ok:
                logger.warning(f'[XYZ Class ] xycenter() V4 failed: {result}')
            return

        response = self.exchange_command('CENTER')
        if response is None:
            logger.warning('[XYZ Class ] xycenter() got no response')

    #----------------------------------------------------------
    # T (Turret) Functions
    #----------------------------------------------------------
    def t_ustep2deg(self, ustep):
        # T config value is usteps per 90 degrees (one turret position)
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        degrees = 90.0 / usteps_per_90deg * ustep
        return degrees

    def t_ustep2pos(self, ustep):
        return int(self.t_ustep2deg(ustep=ustep)/90)+1

    def t_deg2ustep(self, degrees):
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        ustep = int(degrees * usteps_per_90deg / 90.0)
        return ustep

    def t_pos2ustep(self, position):
        """Convert turret position (1-based) to microsteps.
        Uses motorconfig turret positions if available, falls back to 90-degree spacing."""
        usteps = self.motorconfig.turret_position_usteps(position)
        if usteps == 0 and position > 1:
            # Fallback: evenly-spaced positions
            return self.t_deg2ustep(degrees=90*(position-1))
        return usteps

    def thome(self):
        """Home the turret. Returns True on success."""
        if self._use_v4():
            ok, result = self._v4_home_wait({'cmd': 'HOME', 'axis': 'T'},
                                            total_timeout=15)
            logger.info(f'[XYZ Class ] MotorBoard.thome() V4 -> ok={ok} result={result}',
                        extra={'force_error': True})
            result_str = str(result)
            if ok:
                with self._state_lock:
                    self.initial_t_homing_complete = True
                return True
            # "T not present" — board without a turret. Not a failure.
            if 'not present' in result_str.lower():
                return True
            logger.error(f'[XYZ Class ] thome() V4 failed: {result_str}')
            return False

        resp = self.exchange_command('THOME', timeout=15)
        logger.info(f'[XYZ Class ] MotorBoard.thome() -> {resp}', extra={'force_error': True})
        if resp is None:
            logger.error('[XYZ Class ] thome(): no response (timeout or disconnect)')
            return False
        if 'T home successful' in resp:
            with self._state_lock:
                self.initial_t_homing_complete = True
            return True
        # "T not present" is not a failure — board just doesn't have a turret
        if 'not present' in resp.lower():
            return True
        logger.error(f'[XYZ Class ] thome() failed: {resp}')
        return False

    def has_turret(self) -> bool:
        with self._state_lock:
            return self._has_turret

    def has_thomed(self):
        # Note: When the motorboard firmware performs an XYZ homing, it also
        # does a T homing if a turret is present
        with self._state_lock:
            return self.initial_homing_complete or self.initial_t_homing_complete

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------

    def move(self, axis, steps):
        """Move the axis to an absolute position (in usteps) compared to Home.

        This is a low-level function called by move_abs_pos() after limit
        enforcement. Direct callers must ensure steps is within safe range.
        """
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f"Invalid axis {axis!r}")
        if self._use_v4():
            # FW4.0 POS_WRITE takes a 32-bit signed int — no two's-complement
            # trick. Firmware range-checks -2**31 .. 2**31-1.
            if steps > 0x7FFFFFFF:
                # Caller already converted a negative number via the
                # legacy-style two's-complement path above — re-sign it.
                steps = steps - 0x100000000
            if not (-2147483648 <= steps <= 2147483647):
                raise ValueError(f"Steps {steps} out of 32-bit signed range for axis {axis}")
            resp = self.exchange_json({'cmd': 'POS_WRITE', 'axis': axis,
                                        'target': int(steps)})
            if resp is None or resp.get('ok') is not True:
                logger.warning(f'[XYZ Class ] move({axis}, {steps}) V4 no/bad response: {resp}')
            return

        if steps < 0:
            steps += 0x100000000  # two's complement for firmware's unsigned integer format
        if steps > 0xFFFFFFFF:
            raise ValueError(f"Steps {steps} exceeds 32-bit range for axis {axis}")
        response = self.exchange_command('TARGET_W' + axis + str(steps))
        if response is None:
            logger.warning(f'[XYZ Class ] move({axis}, {steps}) got no response')

        # while int(target_pos) != desired_target:
        #     self.exchange_command('TARGET_W' + axis + str(steps))
        #     time.sleep(0.005)
        #     target_pos = int(self.exchange_command('TARGET_R' + axis))

    # Get target position
    def target_pos(self, axis):
        """ Get the target position of an axis"""

        try:
            response = self.exchange_command('TARGET_R' + axis)
            position = int(response)
        except Exception as e:
            logger.warning(f'[XYZ Class ] target_pos({axis}) failed: {e}')
            return None

        if axis == 'Z':
            um = self.z_ustep2um(position)
            return um
        elif (axis == 'X') or (axis == 'Y'):
            um = self.xy_ustep2um(position)
            return um
        elif axis == 'T':
            return self.t_ustep2pos(position)
        else:
            return None

    # Get current position (in um or position for Turret)
    def current_pos(self, axis):
        """Get current position (in um) of axis"""

        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            position = int(response)
        except Exception as e:
            logger.warning(f'[XYZ Class ] current_pos({axis}) failed: {e}')
            return None

        if axis == 'Z':
            um = self.z_ustep2um(position)
            return um
        elif (axis == 'X') or (axis == 'Y'):
            um = self.xy_ustep2um(position)
            return um
        elif axis == 'T':
            return self.t_ustep2pos(position)
        else:
            return None

    # Move to absolute position (in um or degrees for Turret)
    def move_abs_pos(self, axis, pos, overshoot_enabled: bool=True, ignore_limits: bool=False):
        """ Move to absolute position (in um) of axis"""
        # logger.info('move_abs_pos', axis, pos)
        AXES_CONFIG = self.axes_config

        if axis not in AXES_CONFIG:
            raise HardwareError(f"Unsupported axis ({axis})")

        axis_config = AXES_CONFIG[axis]

        if ('limits' in axis_config) and (not ignore_limits):
            axis_limits = axis_config['limits']
            pos = max(pos, axis_limits['min'])
            pos = min(pos, axis_limits['max'])

        steps = axis_config['move_func'](pos)

        if overshoot_enabled and (axis=='Z'): # perform overshoot to always come from one direction
            # get current position
            current = self.current_pos('Z')

            # if the current position is above the new target position
            # and 50um above the height of the backlash
            if current is not None and (current > pos) and (pos > (self.backlash+50)):
                # In process of overshoot
                with self._state_lock:
                    self.overshoot = True
                try:
                    # First overshoot downwards
                    overshoot = self.z_um2ustep(pos-self.backlash) # target minus backlash
                    overshoot = max(1, overshoot)
                    self.move(axis, overshoot)
                    while not self.target_status('Z'):
                        time.sleep(0.02)  # 50Hz — matches motion monitor rate
                finally:
                    # Always clear overshoot flag, even on disconnect/exception
                    with self._state_lock:
                        self.overshoot = False

        self.move(axis, steps)

    # Move by relative distance (in um or degrees for Turret)
    def move_rel_pos(self, axis, um, overshoot_enabled: bool = False):
        """ Move by relative distance (in um for X, Y, Z or position for T) of axis """

        # Read target position in um
        pos = self.target_pos(axis)
        if pos is None:
            logger.warning(f'[XYZ Class ] move_rel_pos({axis}): cannot read position, skipping move')
            return
        self.move_abs_pos(axis, pos+um, overshoot_enabled=overshoot_enabled)

    #----------------------------------------------------------
    # Ramp and Reference Switch Status Register
    #----------------------------------------------------------

    # return True if current and target position are at home.
    def home_status(self, axis):
        """ Return True if axis is in home position"""

        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LIMIT_SW', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                raise RuntimeError(f'LIMIT_SW {axis} failed: {resp}')
            return bool(resp.get('homed'))

        try:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            return bits[31] == '1'
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.home_status('+axis+') inactive')
            raise

    # return True if current position and target position are the same
    def target_status(self, axis):
        """ Return True if axis is at target position"""

        if self._use_v4():
            # FW4.0 LIMIT_SW returns at_target = pos_reached AND vel_zero —
            # the computed field the motion monitor actually wants.
            resp = self.exchange_json({'cmd': 'LIMIT_SW', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                raise RuntimeError(f'LIMIT_SW {axis} failed: {resp}')
            return bool(resp.get('at_target'))

        try:
            payload = 'STATUS_R' + axis
            response = self.exchange_command(payload)
            if response is None:
                raise ValueError("STATUS_R returned None")
            data = int( response )
            bits = format(data, 'b').zfill(32)

            return bits[22] == '1'

        except Exception:
            logger.error('[XYZ Class ] MotorBoard.get_limit_status('+axis+') inactive')
            raise


    # Get all reference status register bits as 32 character string (32-> 0)
    def reference_status(self, axis):
        """ Get all reference status register bits as 32 character string (32-> 0) """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LIMIT_SW', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                raise RuntimeError(f'LIMIT_SW {axis} failed: {resp}')
            # Firmware already includes raw RAMP_STAT in the response for
            # legacy consumers.
            return resp.get('raw', 0)

        try:

            data = int( self.exchange_command('STATUS_R' + axis) )
            # bits = format(data, 'b').zfill(32)

            # data is an integer that represents 4 bytes, or 32 bits,
            # largest bit first
            '''
            bit: 33222222222211111111110000000000
            bit: 10987654321098765432109876543210
            bit: ----------------------*-------**
            '''
            # logger.info(data)
            return data
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.reference_status('+axis+') inactive')
            raise

    def limit_switch_status(self, axis):
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LIMIT_SW', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                logger.warning(f'[XYZ Class ] limit_switch_status({axis}) V4 failed: {resp}')
                return -1, -1
            return (1 if resp.get('left') else 0,
                    1 if resp.get('right') else 0)

        try:
            resp = self.reference_status(axis=axis)
            resp_int = int(resp)
            if resp_int & (1 << 0):
                left = 1
            else:
                left = 0

            if resp_int & (1 << 1):
                right = 1
            else:
                right = 0

        except Exception as e:
            logger.warning(f'[XYZ Class ] limit_switch_status({axis}) failed: {e}')
            left, right = -1, -1

        return left, right


    # ------------------------------------------------------------------
    # Diagnostic commands (firmware v3.0.5+)
    # ------------------------------------------------------------------
    def get_config(self):
        """Send CONFIG and return parsed dict.

        Firmware returns JSON (v3.0.5+) or Python dict repr (older).
        Returns parsed dict, or empty dict on failure.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'CONFIG'})
            if resp is None or resp.get('ok') is not True:
                return {}
            return resp.get('config', {}) or {}

        resp = self.exchange_command('CONFIG')
        if resp is None:
            return {}
        # Take first line (may have trailing newline noise)
        data_str = resp.split('\n')[0].strip() if '\n' in resp else resp.strip()
        import json as _json
        try:
            return _json.loads(data_str)
        except (ValueError, TypeError):
            import ast
            try:
                return ast.literal_eval(data_str)
            except (ValueError, SyntaxError):
                logger.warning(f'[XYZ Class ] get_config(): unparseable response: {data_str[:200]}')
                return {}

    def get_drvstat(self, axis=None):
        """Send DRVSTAT / DRV_STATUS and return parsed driver status.

        Args:
            axis: Optional single axis ('X', 'Y', 'Z', 'T').
                If None, returns status for all axes.

        Returns:
            List of dicts with per-axis fields. LEGACY keys: axis, raw,
            SG, CS, plus flag strings. V4 keys: axis, raw, sg_result,
            cs_actual, stall, ot, otpw, ola, olb, s2ga, s2gb. Both return
            an empty list on failure.
        """
        if self._use_v4():
            payload = {'cmd': 'DRV_STATUS'}
            if axis:
                payload['axis'] = axis
            resp = self.exchange_json(payload)
            if resp is None or resp.get('ok') is not True:
                return []
            if axis:
                # Single-axis response is flat; wrap to match list shape.
                return [{k: v for k, v in resp.items() if k not in ('ok', 'cmd', 'id')}]
            axes_dict = resp.get('axes', {})
            out = []
            for ax, d in axes_dict.items():
                entry = {'axis': ax}
                entry.update(d)
                out.append(entry)
            return out

        cmd = f'DRVSTAT_{axis}' if axis else 'DRVSTAT'
        resp = self.exchange_multiline(cmd, timeout=5, end_markers=['T:'])
        if resp is None:
            # Try single-line for single axis
            if axis:
                resp = self.exchange_command(cmd, timeout=5)
            if resp is None:
                return []

        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        results = []
        for line in lines:
            entry = {'raw_line': line}
            # Parse axis prefix (e.g. "Z: raw=0x...")
            if ':' in line:
                entry['axis'] = line.split(':')[0].strip()
            # Parse raw hex
            import re as _re
            raw_match = _re.search(r'raw=0x([0-9a-fA-F]+)', line)
            if raw_match:
                entry['raw'] = '0x' + raw_match.group(1)
            sg_match = _re.search(r'SG=(\d+)', line)
            if sg_match:
                entry['SG'] = int(sg_match.group(1))
            cs_match = _re.search(r'CS=(\d+)', line)
            if cs_match:
                entry['CS'] = int(cs_match.group(1))
            results.append(entry)
        return results

    def get_motordetect(self):
        """Send MOTOR_DETECT and return parsed motor detection status."""
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'MOTOR_DETECT'})
            if resp is None or resp.get('ok') is not True:
                return []
            axes_dict = resp.get('axes', {})
            out = []
            for ax, d in axes_dict.items():
                out.append({
                    'axis': ax,
                    'detected': bool(d.get('present')),
                    'configured': bool(d.get('configured')),
                })
            return out

        resp = self.exchange_multiline('MOTORDETECT', timeout=5,
                                       end_markers=['T:'])
        if resp is None:
            return []
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        results = []
        for line in lines:
            entry = {'raw_line': line}
            if ':' in line:
                entry['axis'] = line.split(':')[0].strip()
            entry['detected'] = 'detected=True' in line or 'detected=1' in line
            entry['configured'] = 'configured=True' in line or 'configured=1' in line
            results.append(entry)
        return results

    def get_current(self):
        """Send CURRENT and return parsed motor current info."""
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'CURRENT'})
            if resp is None or resp.get('ok') is not True:
                return []
            axes_dict = resp.get('axes', {})
            out = []
            for ax, d in axes_dict.items():
                # Translate V4 field names to legacy upper-case for
                # backward compatibility with existing callers.
                out.append({
                    'axis': ax,
                    'CS_ACTUAL': d.get('cs_actual', 0),
                    'IRUN': d.get('irun', 0),
                    'IHOLD': d.get('ihold', 0),
                    'SG_RESULT': d.get('sg_result', 0),
                    'approx_mA': d.get('approx_mA', 0),
                    'standstill': d.get('standstill', False),
                })
            return out

        resp = self.exchange_multiline('CURRENT', timeout=5,
                                       end_markers=['T:'])
        if resp is None:
            return []
        import re as _re
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        results = []
        for line in lines:
            entry = {'raw_line': line}
            if ':' in line:
                entry['axis'] = line.split(':')[0].strip()
            for key in ('CS_ACTUAL', 'IRUN', 'IHOLD', 'SG_RESULT'):
                m = _re.search(rf'{key}=(\d+)', line)
                if m:
                    entry[key] = int(m.group(1))
            results.append(entry)
        return results

    def get_voltage(self):
        """Send VOLTAGE and return parsed voltage info."""
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'VOLTAGE'})
            if resp is None or resp.get('ok') is not True:
                return {}
            # Legacy callers expect e.g. "24V" key — translate names.
            return {
                '24V': resp.get('v24'),
                '5V':  resp.get('v5'),
                '3V3': resp.get('v3v3'),
                '1V2': resp.get('v1v2'),
                'raw': resp,
            }

        resp = self.exchange_command('VOLTAGE', timeout=5)
        if resp is None:
            return {}
        result = {'raw': resp}
        import re as _re
        for key in ('24V', '5V', '3V3', '1V2'):
            m = _re.search(rf'{key}[=:]\s*([\d.]+|HIGH|LOW|OK)', resp)
            if m:
                result[key] = m.group(1)
        return result

    # ------------------------------------------------------------------
    # Fan control (HiLo discrete or PWM + RPM tach readback)
    # ------------------------------------------------------------------
    #
    # EL-0940 rev 01-04 boards ship with a HiLo fan (BD00C0AWFP driver
    # IC driven via FAN_HILOW GPIO); rev 05+ may ship with a PWM fan
    # reading tach on FANTACH. Firmware auto-detects which hardware is
    # present on both v3.0.x and FW4.0. The driver surfaces a
    # capability probe (`fan_supports_pwm`) so callers can branch UI
    # without hardcoding board rev.
    #
    # Normalized return shape for `get_fan_status`:
    #   {'mode': 'HILO' | 'PWM' | 'NONE',
    #    'state': 'HI' | 'LO' | 'OFF' | None,   # HiLo only
    #    'fan_pct': int | None,                  # PWM only (0-100)
    #    'tach_rpm': int | None,                 # PWM only
    #    'raw': <firmware response, for debugging>}

    def get_fan_status(self):
        """Read fan mode + state + (PWM only) tach RPM.

        V4: `exchange_json({'cmd': 'FAN'})` — firmware returns
        mode/state/pct/tach_rpm in one shot.

        LEGACY: fan info lives in FULLINFO (`FanCntl: HI/LO Speed:X`
        or `FanCntl: PWM Speed: N% Tach: M RPM`). Parse from there,
        prefer cached fullinfo if already populated (avoids an extra
        serial round-trip every poll).

        Returns dict per the module docstring above, or None on
        driver error.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'FAN'})
            if resp is None or resp.get('ok') is not True:
                return None
            mode = resp.get('fan') or resp.get('mode') or 'NONE'
            out = {
                'mode': str(mode).upper(),
                'state': None,
                'fan_pct': resp.get('fan_pct'),
                'tach_rpm': resp.get('tach_rpm'),
                'raw': resp,
            }
            # HiLo state lives in `fan` on FW4.0 when the board has a
            # HiLo controller (fan_hilo.state() returns 'HI'|'LO'|'OFF').
            if out['mode'] in ('HI', 'LO', 'OFF'):
                out['state'] = out['mode']
                out['mode'] = 'HILO'
            return out

        # LEGACY: parse FULLINFO. Prefer cached; only re-fetch if absent.
        raw = None
        with self._state_lock:
            cached = self._fullinfo
        if cached is not None:
            raw = cached.get('_raw')
        if not raw:
            raw = self.exchange_command('FULLINFO', timeout=5) or ''
        return self._parse_legacy_fan_fullinfo(raw)

    @staticmethod
    def _parse_legacy_fan_fullinfo(raw):
        """Parse a v3.0.x FULLINFO response for fan info.

        Firmware emits either:
          `FanCntl: HI/LO   Speed:HI`   (discrete fan)
          `FanCntl: PWM  Speed: 50% Tach: 2345 RPM`  (PWM fan)
        Neither substring present → fan hardware not configured.
        """
        import re as _re
        out = {'mode': 'NONE', 'state': None,
               'fan_pct': None, 'tach_rpm': None, 'raw': raw}
        if not raw:
            return out
        m_hilo = _re.search(r'FanCntl:\s*HI/LO\s+Speed:\s*(HI|LO|OFF)',
                            raw, _re.IGNORECASE)
        if m_hilo:
            out['mode'] = 'HILO'
            out['state'] = m_hilo.group(1).upper()
            return out
        m_pwm = _re.search(
            r'FanCntl:\s*PWM\s+Speed:\s*(\d+)%\s+Tach:\s*(\d+)\s*RPM',
            raw, _re.IGNORECASE)
        if m_pwm:
            out['mode'] = 'PWM'
            out['fan_pct'] = int(m_pwm.group(1))
            out['tach_rpm'] = int(m_pwm.group(2))
        return out

    def fan_supports_pwm(self):
        """True iff the board has a PWM fan (with tach RPM readback).

        Uses `get_fan_status()['mode']`. Result is not cached — the
        caller should cache if polled frequently. Capability is
        hardware-fixed per board, so one call at init is usually
        enough.
        """
        status = self.get_fan_status()
        if not status:
            return False
        return status.get('mode') == 'PWM'

    def set_fan_hilo(self, state):
        """Set HiLo fan to HI / LO / OFF.

        Silent no-op (returns False) if the board doesn't have a HiLo
        fan. Caller can check `get_fan_status()['mode']` first.

        Returns True on firmware-confirmed success, False otherwise.
        """
        state_u = str(state).upper()
        if state_u not in ('HI', 'LO', 'OFF'):
            raise ValueError(f'state must be HI/LO/OFF, got {state!r}')

        if self._use_v4():
            resp = self.exchange_json({'cmd': 'FAN', 'mode': state_u})
            return bool(resp and resp.get('ok') is True)

        # LEGACY: FAN:HI / FAN:LO / FAN:OFF
        resp = self.exchange_command(f'FAN:{state_u}', timeout=5)
        if resp is None:
            return False
        return 'ERROR' not in str(resp).upper()

    def set_fan_pwm(self, pct):
        """Set PWM fan duty cycle 0-100%.

        Silent no-op (returns False) if the board doesn't have a PWM
        fan. Caller can check `fan_supports_pwm()` first.

        Returns True on firmware-confirmed success, False otherwise.
        """
        try:
            pct = int(pct)
        except (TypeError, ValueError):
            raise ValueError(f'pct must be int 0-100, got {pct!r}')
        if not (0 <= pct <= 100):
            raise ValueError(f'pct must be 0-100, got {pct}')

        if self._use_v4():
            resp = self.exchange_json({'cmd': 'FAN', 'mode': pct})
            return bool(resp and resp.get('ok') is True)

        # LEGACY: FANPWM:<pct>
        resp = self.exchange_command(f'FANPWM:{pct}', timeout=5)
        if resp is None:
            return False
        return 'ERROR' not in str(resp).upper()

    def wait_for_position(self, axis, timeout=5.0):
        """Wait until axis reaches target position.

        Polls target_status() at ~100Hz until position is reached or timeout.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            timeout: Maximum wait time in seconds.

        Returns True if position reached, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.target_status(axis):
                    return True
            except Exception:
                pass
            time.sleep(0.01)
        logger.warning(f'[XYZ Class ] wait_for_position({axis}): timed out after {timeout}s')
        return False

    def read_status(self, axis):
        """Read raw RAMP_STAT register value for axis.

        Returns int (32-bit register value), or None on failure.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LIMIT_SW', 'axis': axis})
            if resp is None or resp.get('ok') is not True:
                return None
            return resp.get('raw')

        try:
            resp = self.exchange_command('STATUS_R' + axis)
            if resp is None:
                return None
            return int(resp)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] read_status({axis}) failed: {e}')
            return None

    def get_current_firmware(self):
        """ Returns current version of firmware on Motorboard. LEGACY
        returns the raw text INFO line; V4 returns the structured INFO
        dict. Callers that need both shapes should inspect the return
        type or use fullinfo() for a stable dict shape.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'INFO'})
            if resp is None or resp.get('ok') is not True:
                logger.info('[XYZ Class ] MotorBoard V4 not connected. Unable to check current firmware')
                return
            return resp

        response = self.exchange_command('INFO')
        if not response:
            logger.info('[XYZ Class ] MotorBoard not connected. Unable to check current firmware')
            return
        return response

    def get_axes_config(self):
        return self.axes_config

    def get_axis_limits(self, axis: str):
        AXES_CONFIG = self.axes_config
        if axis not in AXES_CONFIG:
            logger.error(f"[XYZ Class ] MotorBoard.get_axis_limits(): Unsupported axis ({axis})")
            raise HardwareError(f"Unsupported axis ({axis})")

        axis_config = AXES_CONFIG[axis]
        if 'limits' not in axis_config:
            logger.error(f"[XYZ Class ] MotorBoard.get_axis_limits(): No limits defined for axis ({axis})")
            raise HardwareError(f"Axis {axis} does not have defined limits")

        return axis_config['limits']
