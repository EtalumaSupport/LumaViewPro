#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import pathlib
import threading
import time
from lvp_logger import logger

from drivers.serialboard import SerialBoard
from drivers.registry import motor_registry
from drivers.exceptions import HardwareError
from drivers.motorconfig import MotorConfig


class _LegacyAccelProbeFilter(logging.Filter):
    """Drop LVP.serial FIRMWARE ERROR records for the AMAX/DMAX probe
    commands. acceleration_limit() probes per-axis accel/decel registers
    that legacy motor firmware doesn't implement; the response is ERROR,
    the driver catches it and falls back to DEFAULT_ACCELERATION_LIMIT.
    The warning is noise, not a failure. All other FIRMWARE ERROR
    records propagate unchanged."""

    _PROBES = ('AMAXX', 'DMAXX', 'AMAXY', 'DMAXY')

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if 'FIRMWARE ERROR' not in msg:
            return True
        return not any(f'FIRMWARE ERROR: {p}' in msg for p in self._PROBES)


logging.getLogger('LVP.serial').addFilter(_LegacyAccelProbeFilter())


@motor_registry.register('rp2040', priority=100)
class MotorBoard(SerialBoard):
    # ----------------------------------------------------------
    # Initialize connection through microcontroller
    # ----------------------------------------------------------
    def __init__(self, motorconfig_defaults_file: pathlib.Path | None = None, **kwargs):
        self._state_lock = threading.Lock()
        self.overshoot = False
        self._has_turret = False
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self._fullinfo = None
        self._connect_fails = 0
        self._connect_log_suppressed = False

        # Load hardware config (per-unit values from motorconfig.json, with defaults fallback)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path('data/motorconfig_defaults.json')
        self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)

        # Default timeout 5s for regular commands. Long-running commands
        # (HOME, CALIBRATE) pass explicit timeout overrides (H15).
        super().__init__(vid=0x2E8A, pid=0x0005, label='[XYZ Class ]', timeout=5, write_timeout=5)

        # Backward-compatible alias for lock name
        self.thread_lock = self._lock

        # 1. Build cached values from defaults
        self._rebuild_cached_values()
        # 2. Open port, reset firmware, verify connection
        self._initial_connect()
        # 3. Load per-unit config from board, rebuild cache with real values
        self._load_board_config()

    def _rebuild_cached_values(self):
        """Recompute cached values from motorconfig.

        Called at init (with defaults only) and again after
        update_from_board() merges per-unit board data inside connect().

        NOTE: This must NOT call connect(). connect() calls this method
        after update_from_board(), so calling connect() here would recurse
        and attempt to reopen the serial port while it's already open --
        causing PermissionError on Windows. (#610)
        """
        self.backlash = self.motorconfig.antibacklash_um('Z')
        self.axes_config = {
            'Z': {
                'limits': {
                    'min': 0.0,
                    'max': self.motorconfig.travel_limit_um('Z'),
                },
                'move_func': self.z_um2ustep,
            },
            'X': {
                'limits': {
                    'min': 0.0,
                    'max': self.motorconfig.travel_limit_um('X'),
                },
                'move_func': self.xy_um2ustep,
            },
            'Y': {
                'limits': {
                    'min': 0.0,
                    'max': self.motorconfig.travel_limit_um('Y'),
                },
                'move_func': self.xy_um2ustep,
            },
            'T': {'move_func': self.t_pos2ustep},
        }

    def _initial_connect(self):
        """Called once from __init__ to establish the first connection."""
        logger.info('[XYZ Class ] _initial_connect() -- first connection attempt')
        try:
            self.connect()
        except Exception:
            logger.error('[XYZ Class ] _initial_connect() failed')
            raise

    def _load_board_config(self):
        """Read per-unit config from connected board and merge into motorconfig.

        Called once after connect() succeeds. Separate from connect() because
        connect's job is opening the port -- config loading is a post-connect step.
        """
        try:
            board_cfg = self.get_config()
            if board_cfg:
                self.motorconfig.update_from_board(board_cfg)
                self._rebuild_cached_values()
                logger.info(
                    f'[XYZ Class ] Board config merged: model={self.motorconfig.model()}, '
                    f'SN={self.motorconfig.serial_number()}, '
                    f'Z_usteps/mm={self.motorconfig.usteps_per_mm("Z")}'
                )
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

    def connect(self) -> None:
        """Try to connect to the motor controller based on the known VID/PID.

        Idempotent: returns immediately if already connected. Performs
        a legacy port reset (close/reopen) on Windows and primes the
        FULLINFO cache after connecting.
        """
        # Note: _lock is an RLock (from SerialBoard), so re-entrant acquisition
        # by _open_serial, _reset_firmware, exchange_command etc. is safe.
        with self._lock:
            try:
                # Skip if already connected
                if self.driver is not None and self.driver.is_open:
                    logger.debug(
                        f'[XYZ Class ] connect() skipped -- already connected on {self.port}'
                    )
                    return

                logger.info(f'[XYZ Class ] connect() starting on {self.port}')
                self._open_serial()
                logger.info(f'[XYZ Class ] connect() port opened: {self.port}')

                # Legacy port reset: close and reopen to flush USB CDC
                # buffers on Windows. Has existed since original code.
                self.driver.close()
                logger.debug('[XYZ Class ] connect() port closed for reset')
                time.sleep(0.05)  # brief pause for Windows to release port
                self.driver.open()
                logger.debug('[XYZ Class ] connect() port reopened after reset')

                self._connect_fails = 0
                self._connect_log_suppressed = False

                self._reset_firmware()
                info = self.fullinfo()
                with self._state_lock:
                    self._fullinfo = info

                logger.info('[XYZ Class ] Connected to motor controller')
            except Exception as e:
                self._close_driver()
                self._connect_fails += 1
                if self._connect_fails >= 10 and not self._connect_log_suppressed:
                    logger.critical(
                        '[XYZ Class ] MotorBoard.connect() failed 10 times -- suppressing further connect errors (other logging continues)'
                    )
                    self._connect_log_suppressed = True
                if not self._connect_log_suppressed:
                    logger.error(f'[XYZ Class ] MotorBoard.connect() failed: {e}')

    # v3.0 STUB: Motor command builders for JSON Lines protocol
    # When v3.0 is active, commands will use structured JSON format:
    #   {"cmd": "HOME", "axes": ["X", "Y", "Z"]}
    #   {"cmd": "MOVE", "axis": "Z", "target": 12345}
    #   {"cmd": "STATUS", "axis": "Z"}
    #   {"cmd": "SPI", "axis": "Z", "addr": "0x6A", "payload": "0x00"}
    # Currently all commands use the legacy text format.

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

    # ----------------------------------------------------------
    # Informational Functions
    # ----------------------------------------------------------
    def fullinfo(self) -> dict:
        """Send FULLINFO and return parsed model + serial-number dict.

        Returns:
            dict: ``{'model': str, 'serial_number': str, '_raw': str}``.
                Falls back to ``{'model': 'unknown', 'serial_number':
                'unknown'}`` when the response is missing or unparseable.
        """
        info = self.exchange_command('FULLINFO')
        logger.info('[XYZ Class ] MotorBoard.fullinfo(): %s', info, extra={'force_error': True})
        if info is None:
            logger.error('[XYZ Class ] FULLINFO returned None -- board disconnected?')
            return {'model': 'unknown', 'serial_number': 'unknown'}
        # Legacy firmware (pre-FULLINFO) replies with an UNKNOWN_CMD error
        # instead of model/serial. That is an expected capability gap on older
        # units, not a fault -- log at INFO and fall back, rather than an ERROR
        # on every connect, which on a legacy board floods the error log and
        # buries genuine failures (the same noise the VOLTAGE / DRVSTAT /
        # FANSPEED diagnostic probes already suppress on unsupported firmware).
        if 'UNKNOWN_CMD' in info or 'unknown command' in info.lower():
            logger.info(
                '[XYZ Class ] FULLINFO not supported on this firmware; '
                'using model/serial fallback'
            )
            return {'model': 'unknown', 'serial_number': 'unknown'}
        try:
            parts = info.split()
            model = parts[parts.index('Model:') + 1]
            if model[-1] == 'T':
                with self._state_lock:
                    self._has_turret = True
            serial_number = parts[parts.index('Serial:') + 1]
        except (ValueError, IndexError) as e:
            logger.error(f'[XYZ Class ] Failed to parse FULLINFO response: {info!r} ({e})')
            return {'model': 'unknown', 'serial_number': 'unknown'}
        return {
            'model': model,
            'serial_number': serial_number,
            '_raw': info,  # Cached raw response for detect_present_axes()
        }

    def get_microscope_model(self) -> str | None:
        """Return the cached microscope model string from FULLINFO.

        Returns:
            str | None: Model identifier (e.g. 'LS720T'), or None when
                FULLINFO has never completed (disconnected board).
        """
        with self._state_lock:
            info = self._fullinfo
        if info is None:
            # Connection never completed (port held / open() failed) so
            # FULLINFO was never cached. Defensive: the registry's
            # is_connected() gate should keep callers from ever seeing a
            # real MotorBoard with _fullinfo=None, but defense-in-depth
            # demands driver methods never raise on a disconnected
            # instance.
            return None
        return info.get('model')

    def get_serial_number(self) -> str | None:
        """Return the cached serial number string from FULLINFO.

        Served from the FULLINFO response cached at connect; the serial
        number is fixed for the life of a connection, so this never
        re-queries the serial bus.

        Returns:
            str | None: Serial number, or None when FULLINFO has never
                completed (disconnected board).
        """
        with self._state_lock:
            info = self._fullinfo
        if info is None:
            return None
        return info.get('serial_number')

    def detect_present_axes(self) -> list:
        """Detect which axes are present on this board.

        Uses cached FULLINFO from connect() if available, avoiding
        an unnecessary serial round-trip.

        Returns:
            list: Axis letters present, e.g. ``['X', 'Y', 'Z', 'T']`` or
                ``['Z', 'T']``.
        """
        # Use cached fullinfo if available (set during connect)
        with self._state_lock:
            info = self._fullinfo
        if info is not None:
            resp = info.get('_raw', '')
        else:
            resp = self.exchange_command('FULLINFO') or ''
        axes = []
        for axis in ('X', 'Y', 'Z', 'T'):
            if f'{axis} present: True' in resp or f'{axis} present:True' in resp:
                axes.append(axis)
        return axes

    def current_pos_steps(self, axis: str) -> int | None:
        """Get current position in raw microsteps (no unit conversion).

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int | None: Microstep position, or None on failure.
        """
        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] current_pos_steps({axis}) failed: {e}')
            return None

    def target_pos_steps(self, axis: str) -> int | None:
        """Get target position in raw microsteps (no unit conversion).

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int | None: Microstep target, or None on failure.
        """
        try:
            response = self.exchange_command('TARGET_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] target_pos_steps({axis}) failed: {e}')
            return None

    # ----------------------------------------------------------
    # Acceleration control functions
    # ----------------------------------------------------------

    # Cache for acceleration limits -- read once from firmware, reuse thereafter.
    # Invalidated on reconnect via _on_disconnect().
    _accel_cache: dict = None

    # Get single acceleration limit for a specific axis and parameter
    def acceleration_limit(self, axis: str, parameter: str) -> int:
        """Read the firmware acceleration / deceleration limit for an axis.

        Cached after the first read; the cache is invalidated by
        ``_on_disconnect()``. Falls back to a hardcoded default when the
        firmware does not implement the AMAX / DMAX query.

        Args:
            axis: Axis letter ('X' or 'Y').
            parameter: ``'acceleration'`` or ``'deceleration'``.

        Returns:
            int: Maximum register value reported by the firmware (or the
                default when the firmware lacks the query). 0 only on
                input-validation failure that did not raise.

        Raises:
            NotImplementedError: ``axis`` or ``parameter`` is not
                supported.
        """
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return 0

        # Return cached value if available
        cache_key = f'{axis}_{parameter}'
        if self._accel_cache is not None and cache_key in self._accel_cache:
            return self._accel_cache[cache_key]

        parameter_map = {'acceleration': 'A', 'deceleration': 'D'}

        parameter_char = parameter_map[parameter]
        command = f'{parameter_char}MAX{axis}'
        DEFAULT_ACCELERATION_LIMIT = 30000
        using_default = False
        try:
            resp = self.exchange_command(command)

            # In case firmware doesn't support retrieving the acceleration limits
            if resp is None or resp.startswith('ERROR'):
                raise ValueError(f'Firmware returned ERROR for {command}')

            # Extra protection for now in case motorboard responds with a different string that doesnt start with ERROR
            if not resp.isdigit():
                raise ValueError(f'Non-numeric response for {command}: {resp}')

        except Exception:
            resp = DEFAULT_ACCELERATION_LIMIT
            using_default = True

        if using_default:
            logger.debug(
                f'[XYZ Class ] MotorBoard.acceleration_limit({command}): firmware does not support, using default {DEFAULT_ACCELERATION_LIMIT}'
            )
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
            raise NotImplementedError(
                f'Support for acceleration limit on axis {axis} not implemented'
            )

        if parameter not in config['parameters']:
            raise NotImplementedError(
                f'Support for acceleration limit parameter {parameter} not implemented.'
            )

        return True

    def _acceleration_supported_info(self):
        return {'axes': ('X', 'Y'), 'parameters': ('acceleration', 'deceleration')}

    # Get all acceleration limits for all axes and parameters
    def acceleration_limits(self) -> dict[str, dict[str, int]]:
        """Read acceleration + deceleration limits for every supported axis.

        Returns:
            dict: ``{axis: {parameter: int}}`` keyed first by axis letter
                ('X', 'Y') then by ``'acceleration'`` / ``'deceleration'``.
        """
        limits = {}
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            limits[axis] = {}
            for parameter in config['parameters']:
                limits[axis][parameter] = self.acceleration_limit(axis=axis, parameter=parameter)

        return limits

    # Sets the percentage acceleration/deceleration limit (of max) for a single axis/parameter
    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int) -> None:
        """Set acceleration or deceleration as a percentage of the max.

        Resolves the per-axis SPI register address, scales the firmware's
        cached max by ``val_pct``, and writes the result via ``spi_write``.

        Args:
            axis: Axis letter ('X' or 'Y').
            parameter: ``'acceleration'`` or ``'deceleration'``.
            val_pct: Percentage of the maximum (1-100, inclusive).

        Raises:
            NotImplementedError: ``axis`` or ``parameter`` is not
                supported.
            ValueError: ``val_pct`` is outside [1, 100].
        """
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return

        if (val_pct < 1) or (val_pct > 100):
            raise ValueError(
                f'Acceleration limit of {val_pct}% is out of bounds. Must be between 1 and 100.'
            )

        limit = self.acceleration_limit(axis=axis, parameter=parameter)
        setpoint = round(limit * (val_pct / 100))

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

        self.spi_write(axis=axis, addr=SPI_ADDRS[axis][parameter], payload=setpoint)
        logger.info(
            f'[XYZ Class ] MotorBoard.set_acceleration_limit({axis}, {parameter}, {val_pct}%)'
        )

    # Sets the percentage acceleration/deceleration (of max) for all supported axes/parameters
    def set_acceleration_limits(self, val_pct: int) -> None:
        """Apply ``val_pct`` to acceleration + deceleration on every axis.

        Args:
            val_pct: Percentage of the maximum (1-100, inclusive).

        Raises:
            ValueError: ``val_pct`` is outside [1, 100] (raised by
                ``set_acceleration_limit``).
        """
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            for parameter in config['parameters']:
                self.set_acceleration_limit(axis=axis, parameter=parameter, val_pct=val_pct)

    # ----------------------------------------------------------
    # SPI-direct related functions
    # ----------------------------------------------------------
    def spi_read(self, axis: str, addr: int) -> str:
        """Read a TMC motor driver SPI register.

        A dummy ``00`` payload is appended so the firmware accepts the
        request -- the firmware always expects a payload field.

        Args:
            axis: Motor axis ('X', 'Y', 'Z', 'T').
            addr: SPI register address (0x00-0x7F).

        Returns:
            str: Raw response string from the firmware.
        """
        # Add a dummy payload of "00" to the end in order for the firmware to not error out on a read.
        # It is expecting a payload.
        command = f'SPI{axis}0x{addr:02x}00'
        resp = self.exchange_command(command)
        logger.debug(f'[XYZ Class ] MotorBoard.spi_read({axis}, 0x{addr:02x}): {command} -> {resp}')
        return resp

    def spi_write(self, axis: str, addr: int, payload: int | str) -> str:
        """Write to a TMC motor driver SPI register.

        Args:
            axis: Motor axis ('X', 'Y', 'Z', 'T').
            addr: SPI register address (0x00-0x7F; write offset 0x80 added automatically).
            payload: Value to write (decimal integer or string representation).

        Returns:
            str: Raw response string from the firmware.

        Raises:
            ValueError: ``axis`` is invalid or ``addr`` is outside
                [0x00, 0x7F].
        """
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f'Invalid axis {axis!r}')
        if not (0 <= addr <= 0x7F):
            raise ValueError(f'SPI address 0x{addr:02X} out of range [0x00-0x7F]')
        WRITE_OFFSET = 0x80
        write_addr = addr + WRITE_OFFSET
        command = f'SPI{axis}0x{write_addr:02x}{int(payload)}'
        resp = self.exchange_command(command)
        logger.debug(
            f'[XYZ Class ] MotorBoard.spi_write({axis}, 0x{addr:02x}, {payload}): {command} -> {resp}'
        )
        return resp

    # ----------------------------------------------------------
    # Precision mode -- controls motor stop accuracy
    # ----------------------------------------------------------

    # TMC5072 VSTOP register addresses per axis.
    # VSTOP sets the velocity threshold for declaring "stopped" --
    # lower = more accurate final position, slightly slower settle.
    _VSTOP_ADDR = {
        'X': 0x2B,  # VSTOP_M1 on XY chip
        'Y': 0x4B,  # VSTOP_M2 on XY chip
        'Z': 0x4B,  # VSTOP_M2 on ZT chip
        'T': 0x2B,  # VSTOP_M1 on ZT chip
    }
    # Loose stop threshold (VSTOP=1000): fast moves with overshoot
    # tolerance. Used only during AF coarse passes for search speed.
    # Faster than _VSTOP_PRECISION but the final stop position drifts.
    _VSTOP_LOW_PRECISION = 1000
    # Tight stop threshold (VSTOP=100): accurate final position. This
    # is the resting default for normal operation -- motorconfig.py
    # writes vstop=100 to the Z axis at boot, and AF restores ON at
    # every exit path. Anything outside AF should be in this mode.
    _VSTOP_PRECISION = 100

    def set_precision_mode(self, axis: str, enabled: bool) -> None:
        """Set motor precision mode for an axis.

        Precision mode (enabled=True) is the resting default for all
        normal operation -- motorconfig.py writes the precise VSTOP
        threshold to the chip at boot. AF temporarily drops to OFF for
        its coarse passes (search speed) and restores ON for the fine
        pass and all exit paths.

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning (the resting default),
                False for the loose threshold used during AF coarse.
        """
        if axis not in self._VSTOP_ADDR:
            logger.warning(f'[XYZ Class ] set_precision_mode: invalid axis {axis}')
            return
        vstop = self._VSTOP_PRECISION if enabled else self._VSTOP_LOW_PRECISION
        addr = self._VSTOP_ADDR[axis]
        self.spi_write(axis, addr, str(vstop))
        logger.info(f'[XYZ Class ] {axis} precision_mode={enabled} (VSTOP={vstop})')

    # ----------------------------------------------------------
    # Z (Focus) Functions
    # Stock actuator = 0.30 mm pitch.  (1 rev/0.30 mm) x (200 steps/rev) x (256 usteps/step) = 170667 ustep/mm
    # ----------------------------------------------------------
    def z_ustep2um(self, ustep: int) -> float:
        """Convert Z-axis microsteps to micrometers.

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        um = ustep * 1000 / usteps_per_mm
        return um

    def z_um2ustep(self, um: float) -> int:
        """Convert Z-axis micrometers to microsteps.

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def zhome(self) -> bool:
        """Home the objective.

        Returns:
            bool: True on successful Z homing.

        Raises:
            HardwareError: No response from the motor board (timeout or
                disconnect), or firmware reported a homing failure.
        """
        resp = self.exchange_command('ZHOME', timeout=15)
        logger.info(f'[XYZ Class ] MotorBoard.zhome() -> {resp}')
        if resp is None:
            raise HardwareError('zhome(): no response from motor board (timeout or disconnect)')
        success = 'successful' in resp.lower() or 'complete' in resp.lower()
        if not success:
            raise HardwareError(f'zhome(): firmware error: {resp}')
        return True

    # ----------------------------------------------------------
    # XY Stage Functions
    # Stock actuator = 2.54mm pitch.  (1 rev/2.540 mm) x (200 steps/rev) x (256 usteps/step) = 20157 ustep/mm
    # ----------------------------------------------------------

    def xy_ustep2um(self, ustep: int) -> float:
        """Convert XY microsteps to micrometers.

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        um = ustep * 1000 / usteps_per_mm
        return um

    def xy_um2ustep(self, um: float) -> int:
        """Convert XY micrometers to microsteps.

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def home(self) -> bool:
        """Send HOME to firmware and home every axis the board has.

        The firmware's xyzhome routine homes Z, then T, then attempts X/Y.
        On a full XYZ(T) board, the response is 'XYZ home complete'. On a
        Z-only board (LS820 bench), Z (and T if present) get homed first
        and the firmware then returns 'ERROR: X not present' -- the home
        DID succeed for the axes the board has, so this counts as
        success. Real failures (no response, hardware error, or partial
        home aborted by Z/T error) raise HardwareError.

        Returns:
            bool: True on full or partial success.

        Raises:
            HardwareError: No response from the motor board (timeout or
                disconnect), or firmware reported a homing failure.
        """
        resp = self.exchange_command('HOME', timeout=30)
        logger.info(f'[XYZ Class ] MotorBoard.home() -> {resp}', extra={'force_error': True})
        if resp is None:
            raise HardwareError('home(): no response from motor board (timeout or disconnect)')
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
        raise HardwareError(f'home(): firmware error: {resp}')

    def has_homed(self) -> bool:
        """Whether the board has completed an initial XY/Z home cycle.

        Returns:
            bool: True if ``home()`` previously succeeded since
                connect / disconnect.
        """
        with self._state_lock:
            return self.initial_homing_complete

    def xycenter(self) -> None:
        """Move the XY stage to centre (home + objective home included).

        Sends the firmware ``CENTER`` command. Logs a warning on no
        response.
        """
        logger.info('[XYZ Class ] MotorBoard.xycenter()')
        response = self.exchange_command('CENTER')
        if response is None:
            logger.warning('[XYZ Class ] xycenter() got no response')

    # ----------------------------------------------------------
    # T (Turret) Functions
    # ----------------------------------------------------------
    def t_ustep2deg(self, ustep: int) -> float:
        """Convert turret microsteps to degrees.

        Args:
            ustep: Microstep count.

        Returns:
            float: Rotation in degrees.
        """
        # T config value is usteps per 90 degrees (one turret position)
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        degrees = 90.0 / usteps_per_90deg * ustep
        return degrees

    def t_ustep2pos(self, ustep: int) -> int:
        """Convert turret microsteps to a 1-based position number.

        Args:
            ustep: Microstep count.

        Returns:
            int: Turret position (1, 2, 3, 4 ...).
        """
        return int(self.t_ustep2deg(ustep=ustep) / 90) + 1

    def t_deg2ustep(self, degrees: float) -> int:
        """Convert turret degrees to microsteps.

        Args:
            degrees: Rotation in degrees.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        ustep = int(degrees * usteps_per_90deg / 90.0)
        return ustep

    def t_pos2ustep(self, position: int) -> int:
        """Convert turret position (1-based) to microsteps.

        Uses motorconfig turret positions if available, falls back to
        90-degree spacing.

        Args:
            position: Turret position (1-based).

        Returns:
            int: Microstep count for that position.
        """
        usteps = self.motorconfig.turret_position_usteps(position)
        if usteps == 0 and position > 1:
            # Fallback: evenly-spaced positions
            return self.t_deg2ustep(degrees=90 * (position - 1))
        return usteps

    def thome(self) -> bool:
        """Home the turret.

        Returns:
            bool: True on successful turret homing, or when the board
                reports the turret is not present (Z-only boards).

        Raises:
            HardwareError: No response from the motor board (timeout or
                disconnect), or firmware reported a homing failure.
        """
        resp = self.exchange_command('THOME', timeout=15)
        logger.info(f'[XYZ Class ] MotorBoard.thome() -> {resp}', extra={'force_error': True})
        if resp is None:
            raise HardwareError('thome(): no response from motor board (timeout or disconnect)')
        if 'T home successful' in resp:
            with self._state_lock:
                self.initial_t_homing_complete = True
            return True
        # "T not present" is not a failure -- board just doesn't have a turret
        if 'not present' in resp.lower():
            return True
        raise HardwareError(f'thome(): firmware error: {resp}')

    def has_turret(self) -> bool:
        """Whether the connected board has a turret installed.

        Set by ``fullinfo()`` based on the firmware-reported model
        suffix.

        Returns:
            bool: True when the model name ends in ``T``.
        """
        with self._state_lock:
            return self._has_turret

    def has_thomed(self) -> bool:
        """Whether the turret has been homed (or implicitly homed by HOME).

        Returns:
            bool: True if either ``home()`` or ``thome()`` previously
                succeeded since connect / disconnect.
        """
        # Note: When the motorboard firmware performs an XYZ homing, it also
        # does a T homing if a turret is present
        with self._state_lock:
            return self.initial_homing_complete or self.initial_t_homing_complete

    # ----------------------------------------------------------
    # Motion Functions
    # ----------------------------------------------------------

    def move(self, axis: str, steps: int) -> None:
        """Move an axis to an absolute microstep position relative to home.

        This is a low-level helper called by ``move_abs_pos()`` after
        limit enforcement. Direct callers must ensure ``steps`` is within
        safe range.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            steps: Target absolute microstep position. Negative values
                are wrapped to two's-complement so the firmware accepts
                them as unsigned 32-bit integers.

        Raises:
            ValueError: ``axis`` is invalid or ``steps`` exceeds the
                32-bit range.
        """
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f'Invalid axis {axis!r}')
        if steps < 0:
            steps += 0x100000000  # two's complement for firmware's unsigned integer format
        if steps > 0xFFFFFFFF:
            raise ValueError(f'Steps {steps} exceeds 32-bit range for axis {axis}')
        response = self.exchange_command('TARGET_W' + axis + str(steps))
        if response is None:
            logger.warning(f'[XYZ Class ] move({axis}, {steps}) got no response')

        # while int(target_pos) != desired_target:
        #     self.exchange_command('TARGET_W' + axis + str(steps))
        #     time.sleep(0.005)
        #     target_pos = int(self.exchange_command('TARGET_R' + axis))

    # Get target position
    def target_pos(self, axis: str) -> float | int | None:
        """Get the target position of an axis in user units.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            float | int | None: Microns for X/Y/Z, 1-based position for
                T, or None on failure.
        """

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
    def current_pos(self, axis: str) -> float | int | None:
        """Get the current position of an axis in user units.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            float | int | None: Microns for X/Y/Z, 1-based position for
                T, or None on failure.
        """

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
    def move_abs_pos(
        self, axis: str, pos: float, overshoot_enabled: bool = True, ignore_limits: bool = False
    ) -> None:
        """Move an axis to an absolute position in user units.

        For Z, when ``overshoot_enabled`` is True the move first travels
        below the target by ``backlash`` microns and then climbs back
        up so backlash is always taken in the same direction.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            pos: Target absolute position. Microns for X/Y/Z, 1-based
                position for T.
            overshoot_enabled: When True, apply Z backlash compensation
                if the target is sufficiently below the current
                position. Ignored for non-Z axes.
            ignore_limits: When True, skip the configured min/max
                clamping. Use only when caller has explicit knowledge
                that the bare hardware limits are safe.

        Raises:
            HardwareError: ``axis`` is not in ``axes_config``.
        """
        # logger.info('move_abs_pos', axis, pos)
        AXES_CONFIG = self.axes_config

        if axis not in AXES_CONFIG:
            raise HardwareError(f'Unsupported axis ({axis})')

        axis_config = AXES_CONFIG[axis]

        if ('limits' in axis_config) and (not ignore_limits):
            axis_limits = axis_config['limits']
            pos = max(pos, axis_limits['min'])
            pos = min(pos, axis_limits['max'])

        steps = axis_config['move_func'](pos)

        if overshoot_enabled and (
            axis == 'Z'
        ):  # perform overshoot to always come from one direction
            # get current position
            current = self.current_pos('Z')

            # if the current position is above the new target position
            # and 50um above the height of the backlash
            if current is not None and (current > pos) and (pos > (self.backlash + 50)):
                # In process of overshoot
                with self._state_lock:
                    self.overshoot = True
                try:
                    # First overshoot downwards
                    overshoot = self.z_um2ustep(pos - self.backlash)  # target minus backlash
                    overshoot = max(1, overshoot)
                    self.move(axis, overshoot)
                    while not self.target_status('Z'):
                        time.sleep(0.02)  # 50Hz -- matches motion monitor rate
                finally:
                    # Always clear overshoot flag, even on disconnect/exception
                    with self._state_lock:
                        self.overshoot = False

        self.move(axis, steps)

    # Move by relative distance (in um or degrees for Turret)
    def move_rel_pos(self, axis: str, um: float, overshoot_enabled: bool = False) -> None:
        """Move an axis by a relative offset in user units.

        Reads the current target, adds ``um``, and dispatches an
        absolute move. Logs a warning and skips when the target read
        fails.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            um: Offset to apply. Microns for X/Y/Z, position-count
                offset for T.
            overshoot_enabled: When True, apply Z backlash compensation
                during the underlying absolute move.
        """

        # Read target position in um
        pos = self.target_pos(axis)
        if pos is None:
            logger.warning(
                f'[XYZ Class ] move_rel_pos({axis}): cannot read position, skipping move'
            )
            return
        self.move_abs_pos(axis, pos + um, overshoot_enabled=overshoot_enabled)

    # ----------------------------------------------------------
    # Ramp and Reference Switch Status Register
    # ----------------------------------------------------------

    # return True if current and target position are at home.
    def home_status(self, axis: str) -> bool:
        """Return True if the axis is in the home position.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            bool: True when the firmware reports the axis at home.

        Raises:
            Exception: Re-raises any error from the STATUS_R query.
        """

        # logger.info('[XYZ Class ] MotorBoard.home_status('+axis+')')
        try:
            data = int(self.exchange_command('STATUS_R' + axis))
            bits = format(data, 'b').zfill(32)

            return bits[31] == '1'
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.home_status(' + axis + ') inactive')
            raise

    def motor_stop(self) -> bool:
        """LVP-A-1 followup: stop all motors via STOP, with field-firmware fallback.

        Field firmware (e.g. EL-0940-02 2024-09-10) does not implement
        the ``STOP`` command and replies ``ERROR: command 'STOP' not
        found``. Newer firmware (2025-onward) accepts STOP as the
        emergency-stop command (sets target=actual on every axis).

        Behavior:
        - First call: send STOP. Inspect the response -- if it contains
          ``not found`` or starts with ``ERROR``, cache the firmware
          as unsupported and return False (silent skip on future
          calls). Otherwise return True.
        - Subsequent calls: skip the wire entirely if cached
          unsupported.

        The caller's shutdown is unaffected when STOP isn't supported:
        the host is about to disconnect anyway, and v3.0.x firmware
        latches its current state when the host stops issuing commands.
        Returning False lets the caller know the stop didn't actually
        execute (useful for tests / diagnostics).

        Idempotent + safe to call concurrently with other operations
        (per SerialBoard's exchange_command lock).
        """
        # Cached "unsupported" -- silently skip the wire (and skip the
        # FIRMWARE ERROR warning that exchange_command would emit).
        if getattr(self, '_stop_supported', None) is False:
            return False
        # expect_unsupported=True suppresses the FIRMWARE ERROR warning
        # on this first probe -- the unsupported case is handled
        # immediately below by caching _stop_supported=False and
        # logging an informational message instead.
        resp = self.exchange_command('STOP', expect_unsupported=True)
        resp_str = str(resp) if resp is not None else ''
        if 'not found' in resp_str or resp_str.startswith('ERROR'):
            self._stop_supported = False
            logger.info(
                '[XYZ Class ] motor_stop: firmware does not support '
                f'STOP command (firmware date '
                f'{getattr(self, "firmware_date", "unknown")}); '
                'caching capability and silently skipping future STOP '
                'attempts. Motors latch on host disconnect.'
            )
            return False
        self._stop_supported = True
        return True

    # return True if current position and target position are the same
    def target_status(self, axis: str) -> bool:
        """Return True if the axis has reached its target position.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            bool: True when the firmware reports current == target.

        Raises:
            HardwareError: No response from the motor board (timeout or
                disconnect); re-raised after logging.
        """

        # logger.info('[XYZ Class ] MotorBoard.target_status('+axis+')')
        try:
            payload = 'STATUS_R' + axis
            response = self.exchange_command(payload)
            if response is None:
                raise HardwareError(
                    f'target_status({axis}): no response from motor board '
                    '(STATUS_R returned None -- timeout or disconnect)'
                )
            data = int(response)
            bits = format(data, 'b').zfill(32)

            return bits[22] == '1'

        except Exception:
            logger.error('[XYZ Class ] MotorBoard.target_status(' + axis + ') inactive')
            raise

    # Get all reference status register bits as 32 character string (32-> 0)
    def reference_status(self, axis: str) -> int:
        """Read the raw STATUS register for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int: 32-bit register value as returned by the firmware.

        Raises:
            Exception: Re-raises any error from the STATUS_R query.
        """
        try:
            data = int(self.exchange_command('STATUS_R' + axis))
            # bits = format(data, 'b').zfill(32)

            # data is an integer that represents 4 bytes, or 32 bits,
            # largest bit first
            """
            bit: 33222222222211111111110000000000
            bit: 10987654321098765432109876543210
            bit: ----------------------*-------**
            """
            # logger.info(data)
            return data
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.reference_status(' + axis + ') inactive')
            raise

    def limit_switch_status(self, axis: str) -> tuple[int, int]:
        """Read the left + right limit switch state for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            tuple[int, int]: ``(left, right)`` where each value is 1
                (engaged), 0 (clear), or -1 on read failure.
        """
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
    def get_config(self) -> dict:
        """Send CONFIG and return the parsed per-board configuration.

        Firmware returns JSON (v3.0.5+) or Python dict repr (older).

        Returns:
            dict: Parsed configuration, or an empty dict on failure.
        """
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

    def get_drvstat(self, axis: str | None = None) -> list:
        """Send DRVSTAT and return parsed driver status.

        Args:
            axis: Optional single axis ('X', 'Y', 'Z', 'T').
                If None, returns status for all axes.

        Returns:
            list: One dict per axis with keys ``axis``, ``raw`` (hex
                string), ``SG`` (int), ``CS`` (int), plus flag strings
                from the firmware. Empty list on failure.
        """
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

    def get_motordetect(self) -> list:
        """Send MOTORDETECT and return parsed motor detection status.

        Returns:
            list: One dict per axis with keys ``axis``, ``detected``
                (bool), ``configured`` (bool), and ``raw_line``. Empty
                list on failure.
        """
        resp = self.exchange_multiline('MOTORDETECT', timeout=5, end_markers=['T:'])
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

    def get_current(self) -> list:
        """Send CURRENT and return parsed motor current info.

        Returns:
            list: One dict per axis with keys ``axis``, ``CS_ACTUAL``,
                ``IRUN``, ``IHOLD``, ``SG_RESULT`` (all int), and
                ``raw_line``. Empty list on failure.
        """
        resp = self.exchange_multiline('CURRENT', timeout=5, end_markers=['T:'])
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

    def get_voltage(self) -> dict:
        """Send VOLTAGE and return parsed voltage rail info.

        Returns:
            dict: ``raw`` plus per-rail keys (``24V``, ``5V``, ``3V3``,
                ``1V2``) when matched. Empty dict on failure.
        """
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

    def wait_for_position(self, axis: str, timeout: float = 5.0) -> bool:
        """Wait until an axis reaches its target position.

        Polls target_status() at ~100Hz until position is reached or timeout.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            timeout: Maximum wait time in seconds.

        Returns:
            bool: True if the position was reached, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.target_status(axis):
                    return True
            except Exception as e:
                logger.debug(
                    '[XYZ Class ] wait_for_position(%s): target_status poll '
                    'raised; continuing to poll: %s: %s',
                    axis,
                    type(e).__name__,
                    e,
                )
            time.sleep(0.01)
        logger.warning(f'[XYZ Class ] wait_for_position({axis}): timed out after {timeout}s')
        return False

    def read_status(self, axis: str) -> int | None:
        """Read raw STATUS register value for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int | None: 32-bit register value, or None on failure.
        """
        try:
            resp = self.exchange_command('STATUS_R' + axis)
            if resp is None:
                return None
            return int(resp)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] read_status({axis}) failed: {e}')
            return None

    def get_current_firmware(self) -> str | None:
        """Return the current motor-controller firmware identification.

        Returns:
            str | None: Multi-line response from the INFO command (e.g.
                ``Etaluma Motor Controller Board <BOARD>\\nFirmware:
                <DATE>``), or None when the board did not respond.
        """
        response = self.exchange_command('INFO')
        if not response:
            logger.info('[XYZ Class ] MotorBoard not connected. Unable to check current firmware')
            return
        return response

    def get_axes_config(self) -> dict:
        """Return the per-axis config (limits + unit-conversion func).

        Returns:
            dict: Axis-letter-keyed configuration. Each value contains
                ``limits`` (when applicable) and ``move_func`` (the
                user-units-to-microsteps conversion).
        """
        return self.axes_config

    def get_axis_limits(self, axis: str) -> dict | None:
        """Return the configured min/max travel limits for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            dict: ``{'min': float, 'max': float}`` in axis user units,
                or ``None`` if the axis has no configured limits (the
                turret axis T is the typical "no limits" case -- it
                rotates freely with no software-enforced bounds).

        Raises:
            HardwareError: ``axis`` is not a supported axis at all
                (programmer error, not a configuration variant).
        """
        AXES_CONFIG = self.axes_config
        if axis not in AXES_CONFIG:
            logger.error(f'[XYZ Class ] MotorBoard.get_axis_limits(): Unsupported axis ({axis})')
            raise HardwareError(f'Unsupported axis ({axis})')

        axis_config = AXES_CONFIG[axis]
        if 'limits' not in axis_config:
            # Expected for axes without software-enforced bounds (T axis
            # is the canonical case). Callers must handle None.
            return None

        return axis_config['limits']

    # ------------------------------------------------------------------
    # Diagnostic queries -- firmware-version-gated
    # ------------------------------------------------------------------
    # VOLTAGE / DRVSTAT_<axis> / FANSPEED / FAN:<duty> are diagnostic
    # commands added in firmware revisions after 2024-09-10. Older
    # firmware responds with `ERROR: command 'X' not found:` and the
    # driver returns None / False instead of leaking that response to
    # callers. Firmware-shape knowledge stays here so diagnostic
    # callers (TSR, future REST diagnostic endpoint) need not parse
    # raw firmware responses.

    def _diagnostic_query(self, command: str) -> str | None:
        """Send a capability-gated diagnostic command.

        Returns the raw response string, or None when the firmware
        rejected the command with an `ERROR` prefix (i.e. the legacy
        firmware does not implement this query). Callers should treat
        None as "INCONCLUSIVE -- firmware does not support this
        diagnostic," NOT as a fault.

        expect_unsupported=True is passed to exchange_command so that
        firmware-not-found responses on legacy firmware (2024-09-10
        and earlier) are downgraded to DEBUG, matching the motor_stop
        capability probe pattern. Without this opt-in, the WARNING
        fires from serialboard.exchange_command BEFORE this helper
        gets to swallow the ERROR response, leaking noise into the
        user-visible log on every TSR run against legacy firmware.
        """
        resp = self.exchange_command(command, expect_unsupported=True)
        if resp is None:
            return None
        if resp.startswith('ERROR'):
            logger.debug(
                f'[XYZ Class ] MotorBoard.{command} not supported by '
                f'connected firmware (response: {resp!r})'
            )
            return None
        return resp

    def read_voltages(self) -> dict[str, float | None] | None:
        """Read power-rail voltage tolerance diagnostic.

        Returns a dict mapping rail label ('5V', '3.3V', '1.2V', '24V')
        to the measured voltage in volts, or None for any rail whose
        firmware reading was non-numeric (e.g. 'OK', 'N/A', 'MISSING').
        Returns None for the whole call when the firmware does not
        support the VOLTAGE command (legacy firmware predating
        diagnostic queries). Callers should distinguish:
            None              -> INCONCLUSIVE: firmware does not support
            {rail: None, ...} -> INCONCLUSIVE: per-rail unparseable
            {rail: float}     -> measurement available
        """
        raw = self._diagnostic_query('VOLTAGE')
        if raw is None:
            return None
        # Firmware response shape: '24V=OK 5V=5.18 3V3=3.31 1V2=1.24'
        # (or 'N/A' / 'MISSING' / 'ERROR' in the value slot).
        # Normalize '3V3' -> '3.3V' and '1V2' -> '1.2V' so caller
        # comparison against VOLTAGE_NOMINAL keys lines up.
        rail_rename = {'3V3': '3.3V', '1V2': '1.2V'}
        non_numeric = {'OK', 'N/A', 'MISSING', 'ERROR'}
        rails: dict[str, float | None] = {}
        for token in raw.split():
            if '=' not in token:
                continue
            key, _, value = token.partition('=')
            label = rail_rename.get(key, key)
            if value in non_numeric:
                rails[label] = None
                continue
            try:
                rails[label] = float(value.rstrip('V'))
            except ValueError:
                rails[label] = None
        return rails

    def read_drv_status(self, axis: str) -> int | None:
        """Read TMC5072 DRV_STATUS register for an axis.

        Returns the raw 32-bit register value as int (caller decodes
        bits), or None if firmware does not support DRVSTAT_<axis>.
        Axis must be one of 'X', 'Y', 'Z', 'T'.
        """
        axis = axis.upper()
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f'Invalid axis: {axis!r}')
        raw = self._diagnostic_query(f'DRVSTAT_{axis}')
        if raw is None:
            return None
        try:
            return int(raw.strip().split()[0], 0)
        except (ValueError, IndexError):
            logger.warning(f'[XYZ Class ] DRVSTAT_{axis} unparseable: {raw!r}')
            return None

    def read_fanspeed(self) -> int | None:
        """Read fan tachometer RPM.

        Returns RPM as int (0 if tachometer wire not installed),
        or None if firmware does not support FANSPEED.
        """
        raw = self._diagnostic_query('FANSPEED')
        if raw is None:
            return None
        try:
            return int(raw.strip().split()[0])
        except (ValueError, IndexError):
            logger.warning(f'[XYZ Class ] FANSPEED unparseable: {raw!r}')
            return None

    def set_fan_duty(self, duty_pct: int) -> bool:
        """Set fan PWM duty cycle (0..100). Returns True if firmware
        accepted the command, False if firmware does not support
        FAN:<duty>.
        """
        if not 0 <= duty_pct <= 100:
            raise ValueError(f'Fan duty must be 0..100, got {duty_pct}')
        resp = self.exchange_command(f'FAN:{duty_pct}')
        if resp is None:
            return False
        if resp.startswith('ERROR'):
            logger.debug(
                f'[XYZ Class ] FAN:{duty_pct} not supported by '
                f'connected firmware (response: {resp!r})'
            )
            return False
        return True
