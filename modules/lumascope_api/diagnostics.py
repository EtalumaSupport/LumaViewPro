# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""DiagnosticsAPI -- sub-API for hardware diagnostic probes.

Stateless probes for camera / motor / LED hardware. No persistent
state -- per-call probes.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import pathlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lvp_logger import log_dir, logger
from modules.finite_number import refuse_outside, refuse_unless_wait_s
from modules.api_surface import Admission, ProgressCallback, api, api_fields
from modules.exceptions import (
    ArgumentRefusedError,
    CaptureFailedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from modules.axes import Axis

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope

# Motor-board verbs that move a motor, stop one, or rewrite a position
# register. Sent raw they change the hardware behind MotionAPI, which then
# reports axes and a turret slot that are no longer true; each has a
# MotionAPI member that does the same thing and keeps the record. Matched
# as the firmware dispatches them: it upper-cases and strips the command,
# compares the homes and STOP whole, and the shipping firmware finds
# ACTUAL_W / TARGET_W / SPI anywhere in the command.
_MOTOR_VERBS_WHOLE = frozenset({'HOME', 'CENTER', 'ZHOME', 'THOME', 'STOP'})
_MOTOR_VERBS_ANYWHERE = ('ACTUAL_W', 'TARGET_W')
# The one SPI form carried: an axis, a two-digit address below 0x80 (the
# TMC5072 read half; bit 7 set is a register write) and a decimal payload.
_SPI_READ = re.compile(r'SPI[A-Z]0X([0-7][0-9A-F])\d+')

# What the diagnostics channel answers in place of a board's reply. They
# come back as strings so a report can write them where the reply would
# have gone; anything that times, parses or counts a reply tells them from
# one with ``is_board_reply``.
# What the camera benchmarks take: generous for any characterization, and
# short of a lane wait the host cannot hold (a huge one overflowed it).
_BENCHMARK_CYCLES = (1, 10_000)
# A test of no frames received none and would answer passed.
_BANDWIDTH_FRAMES = (1, 100_000)
_BENCHMARK_DELAY_S = (0, 60)
_BENCHMARK_SLOW_S = (0, 600)
_PROBE_DURATION_S = (0, 3_600)


@api_fields(
    'instance_id', 'present', 'service', 'provider', 'inf', 'version', 'date', 'problem_code'
)
@dataclass(frozen=True)
class UsbDeviceBinding:
    """One device of a USB identity and the driver Windows has bound to it.

    Attributes:
        instance_id: Windows' instance ID: the identity, a backslash, then
            the device's serial or port.
        present: False for a device seen before on another port or at
            another time, which keeps its binding until it is matched afresh.
        service: The bound driver's service (``WinUSB``, ``libusbK``,
            ``libusb0``); None when no driver is bound, and the package
            fields with it.
        provider: The bound package's provider.
        inf: The bound package's published INF (``oem12.inf``).
        version: The bound package's version.
        date: The bound package's date.
        problem_code: Device Manager's problem code; None for none.
    """

    instance_id: str
    present: bool
    service: str | None
    provider: str | None
    inf: str | None
    version: str | None
    date: str | None
    problem_code: int | None


@api_fields('hardware_id', 'name', 'needs', 'usable', 'devices')
@dataclass(frozen=True)
class UsbIdentityBindings:
    """A USB identity of a Lumascope Classic and each of its devices' drivers.

    Attributes:
        hardware_id: ``USB\\VID_04B4&PID_8613`` (the bootloader) or
            ``USB\\VID_04B4&PID_EA17`` (the running firmware).
        name: Its name in Device Manager.
        needs: The drivers LumaViewPro can use it on, lower case.
        usable: False when a present device is on none of them; a device
            seen before does not count, since it is matched afresh when
            plugged.
        devices: Every device of it, present or seen before.
    """

    hardware_id: str
    name: str
    needs: tuple[str, ...]
    usable: bool
    devices: tuple[UsbDeviceBinding, ...]


@api_fields('transport', 'link_speed', 'link_speed_unit', 'packet_size_bytes', 'inter_packet_delay')
@dataclass(frozen=True)
class CameraLink:
    """The camera's link to the host, read live; each field None where the
    camera does not report it.

    Attributes:
        transport: ``'USB3'``, ``'GigE'`` or ``'USB2'``; the SDK's own name
            when it is none of these.
        link_speed: The negotiated speed, in ``link_speed_unit``, never
            converted: GenICam's DeviceLinkSpeed has a unit that varies by
            Basler model, and libusb reports the FX2's in Mbps. A speed whose
            camera declares no unit is None with its unit, a number no caller
            could read.
        link_speed_unit: The unit the camera declares for ``link_speed``.
        packet_size_bytes: GigE's stream packet size.
        inter_packet_delay: GigE's inter-packet delay, in the camera's ticks.
    """

    transport: str | None
    link_speed: float | None
    link_speed_unit: str | None
    packet_size_bytes: int | None
    inter_packet_delay: int | None


class MotorCommandSet(enum.StrEnum):
    """The motor board's text command set, as ``get_motor_info().command_set`` names it.

    The EL-0940's (INFO, FULLINFO, ACTUAL_R, SPI and the rest). A board
    with no text command channel (the TMCM-6110, which speaks binary
    datagrams), or no board connected, answers None.
    """

    TEXT = 'text'


class LedCommandSet(enum.StrEnum):
    """The LED board's text command set, as ``get_led_info().command_set`` names it.

    INFO predates the v2 firmware; SELFTEST, I2CSCAN and the LEDREAD
    current reads came with it. A board with no text command channel at
    all (the FX2's LED peripheral) answers None.
    """

    LEGACY = 'legacy'
    V2 = 'v2'


@api_fields('model', 'serial_number', 'firmware_version', 'firmware_date', 'command_set')
@dataclass(frozen=True)
class MotorBoardInfo:
    """The motor board's identity, as it reported it at connect.

    Each field is None when it is unknown: no board, a board that did not
    answer, or a reply that could not be read.

    Attributes:
        model: The scope model the board reports (``'LS850T'``), the board's
            own report and not ``channel_identity.model``, which falls back to
            the operator's selection.
        serial_number: The board's serial number.
        firmware_version: The parsed version, or ``ORIGINAL_FIRMWARE``
            (``'original'``) for a board that answered INFO with no version
            string.
        firmware_date: The date INFO carried.
        command_set: ``MotorCommandSet.TEXT`` for a board with a text command
            channel; None for one without (the TMCM-6110, which speaks binary
            datagrams) and for no board.
    """

    model: str | None
    serial_number: str | None
    firmware_version: str | None
    firmware_date: str | None
    command_set: MotorCommandSet | None


@api_fields('firmware_version', 'firmware_date', 'command_set')
@dataclass(frozen=True)
class LedBoardInfo:
    """The LED board's firmware, as it reported it at connect.

    Each field is None when it is unknown: no board, a board that did not
    answer, or one with no firmware of its own (an FX2 scope's LED
    peripheral). Whether a board is connected is ``led_connected``.

    Attributes:
        firmware_version: As ``MotorBoardInfo.firmware_version``.
        firmware_date: As ``MotorBoardInfo.firmware_date``.
        command_set: ``LedCommandSet.V2`` (INFO, SELFTEST, I2CSCAN, LEDREAD),
            ``LedCommandSet.LEGACY`` (INFO only: firmware older than v2), or
            None: no text command channel, as on an FX2 scope, or no board.
    """

    firmware_version: str | None
    firmware_date: str | None
    command_set: LedCommandSet | None


NOT_CONNECTED = 'Board not connected'
NO_REPLY = 'None'  # a single-line read that timed out
NO_RESPONSE = 'No response'  # a multi-line read that returned nothing
ERROR_PREFIX = 'Error: '

# What ``firmware_version`` names for a board that answered INFO with no
# version string: the original firmware, which predates version numbers and
# is told apart by its date. None is kept for a board that did not answer.
ORIGINAL_FIRMWARE = 'original'


def _firmware_identity(driver) -> dict:
    """A board's firmware as the API names it: version and date.

    The driver parses INFO; the original firmware carries a date but no
    version, which the driver records as no version on a board that
    answered. Named here, once, so the support report and a bench verdict
    both say which firmware it was.
    """
    version = driver.firmware_version
    if version is None and driver.firmware_responding:
        version = ORIGINAL_FIRMWARE
    return {'firmware_version': version, 'firmware_date': driver.firmware_date}


def is_board_reply(response: str | list[str] | None) -> bool:
    """True when ``response`` is a board's reply, not the channel's stand-in for one."""
    if not response or response in (NOT_CONNECTED, NO_REPLY, NO_RESPONSE):
        return False
    return not (isinstance(response, str) and response.startswith(ERROR_PREFIX))


def _refuse_motor_verb(command: str) -> None:
    """Raise if ``command`` would move or stop a motor or rewrite a position."""
    verb = command.strip().upper()
    if (
        verb in _MOTOR_VERBS_WHOLE
        or any(v in verb for v in _MOTOR_VERBS_ANYWHERE)
        or ('SPI' in verb and not _SPI_READ.fullmatch(verb))
    ):
        raise ValueError(
            f'{command!r} moves, stops or repositions a motor and is not carried by '
            f'the diagnostics channel: use the motion API (home, move_turret, '
            f'move_xy, move_z, stop_motion), which keeps the axis and turret record'
        )


class DiagnosticsAPI:
    """Diagnostics sub-API. Forwards to Lumascope composition root.

    Every member that transmits runs on its device's lane, so while a run, a
    diagnostic or a home holds the scope it is refused to anyone not acting under
    the holder's taking. The one exception is the camera temperature read,
    which the lane admits whatever holds: it changes nothing a holder
    depends on, and the temperature log keeps running through a run. The
    board commands go on the IO lane through the motion API's dispatcher,
    the lane's one blocking dispatch with a wait bound.
    """

    # Wait bounds past the command's own serial timeout: queue time behind
    # the IO lane's other work, which is a liveness margin, not a budget.
    _BOARD_QUEUE_MARGIN_S = 30.0
    # A grab stop or start has been measured near 11 s on a Pylon body; a
    # cycle is one of each, so this bounds one cycle with room to spare.
    _GRAB_CYCLE_BOUND_S = 30.0

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope

    # --- Camera probes ---
    @api(reads=True, admitted_by=(Admission.LANE_CALL,))
    def get_camera_temperatures_degc(self) -> dict[str, float] | None:
        """Get all camera temperature sensor readings.

        Runs on the camera lane, admitted whatever holds the scope: the read
        sets the camera's temperature selector, which must not interleave
        with another camera write, but a run does not stop it.

        Returns:
            dict[str, float] | None: Each sensor's temperature in degC, by
            the sensor's name, which is the camera's own (a Basler body's
            ``'FpgaCore'``; ``'Device'`` on an IDS body with one sensor). Empty
            only for a camera with no temperature sensor
            (``capabilities.camera_reports_temperature`` False). None when no
            camera is connected, after ``disconnect()`` too: a read with no
            camera to describe.

        Raises:
            CameraReadFailedError: ``'camera_read_failed'``, a camera is
                active and the read failed.
        """
        return self._read_camera_or_none(
            self._get_camera_temperatures_degc_impl, 'get_camera_temperatures_degc'
        )

    def _get_camera_temperatures_degc_impl(self) -> dict[str, float] | None:
        driver = self._scope._camera_driver
        if not driver or not driver.active:
            return None
        return driver.get_all_temperatures()

    @api(reads=True, admitted_by=(Admission.LANE_CALL,))
    def get_camera_link_info(self) -> CameraLink | None:
        """Read the camera's link, live.

        Runs on the camera lane, admitted whatever holds the scope, like the
        temperature read: the reads are node reads that must not interleave
        with a camera write, and a run does not stop them.

        Returns:
            CameraLink | None: The link, each field None where the camera
                does not report it. None when no camera is connected, after
                ``disconnect()`` too.

        Raises:
            CameraReadFailedError: ``'camera_read_failed'``, a field the
                camera reports could not be read.
        """
        return self._read_camera_or_none(self._get_camera_link_info_impl, 'get_camera_link_info')

    def _read_camera_or_none[T](self, impl: Callable[[], T | None], name: str) -> T | None:
        """Run a camera read on the camera lane; None with no camera connected.

        Asked before dispatch, so after ``disconnect()`` -- when the lane
        would refuse -- the read still answers that there is no camera to
        read. A camera that leaves while the read is queued is answered the
        same way at the lane.
        """
        camera = self._scope.camera
        if not self._scope.camera_connected:
            return None
        return camera._dispatch_camera(
            impl,
            name,
            timeout_s=camera._CAMERA_WRITE_TIMEOUT_S,
            override=True,
            satisfied_when_absent=None,
        )

    def _get_camera_link_info_impl(self) -> CameraLink | None:
        driver = self._scope._camera_driver
        if not driver or not driver.active:
            return None
        # Built by name, so a field the drivers' link_info gains or loses
        # fails here rather than going unpublished.
        return CameraLink(**driver.get_link_info())

    @api(reads=True)
    def get_camera_usb_bindings(self) -> tuple[UsbIdentityBindings, ...] | None:
        """Which driver Windows has bound to each USB identity of a Lumascope Classic.

        The FX2 is two USB devices in turn, the bootloader and the running
        firmware, and Windows binds each on its own. Read without elevation,
        the same read the camera's connect checks, so it answers whether or
        not the camera came up -- the case it is for. Takes no lane: it asks
        the host's device list, not the scope.

        Returns:
            tuple[UsbIdentityBindings, ...] | None: The bootloader's, then the
                running firmware's. None where the host binds no driver to a
                USB device (macOS, Linux) and on a simulated scope with no FX2.

        Raises:
            CameraUsbLookupFailedError: Windows' device list could not be read.
        """
        from drivers.usb_driver_binding import (
            DEVICE_NAMES,
            FX2_BOOTLOADER,
            FX2_FIRMWARE,
            USABLE_SERVICES,
            unusable,
        )

        answer = []
        for identity in (FX2_BOOTLOADER, FX2_FIRMWARE):
            devices = self._scope._read_fx2_bindings(identity)
            if devices is None:
                return None
            answer.append(
                UsbIdentityBindings(
                    hardware_id=identity,
                    name=DEVICE_NAMES[identity],
                    needs=tuple(sorted(USABLE_SERVICES[identity])),
                    usable=not unusable(identity, devices),
                    # Built field by field, so a field the driver's record
                    # gains or loses fails here rather than going unpublished.
                    devices=tuple(UsbDeviceBinding(**dataclasses.asdict(d)) for d in devices),
                )
            )
        return tuple(answer)

    @api(in_process=True, engineering=True)
    def run_camera_bandwidth_test(
        self,
        num_frames: int,
        *,
        timeout_s: float = 60.0,
        progress_cb: ProgressCallback | None = None,
    ) -> dict:
        """Run an N-frame camera throughput test through the production capture path.

        Every frame is grabbed through ``ImagingAPI.get_image``, so the
        numbers are those of the path a protocol and the live view capture
        through.

        Args:
            num_frames: Total frames to grab.
            timeout_s: Hard wall-clock cutoff in seconds; the test stops
                early and marks ``passed=False`` if exceeded.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 250 frames.

        Returns:
            dict: num_frames_requested, num_frames_received,
                num_frames_failed (the grabs that raised
                ``CaptureFailedError``), failed_by_reason (those, counted by
                the failure's reason), total_bytes, elapsed_seconds,
                mb_per_second, fps_actual, frame_sizes, errors (the first 20
                failures' words), passed, and resolution (``'WxH'``) and
                pixel_format, as the camera held them, None when unknown.

        Raises:
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                ``timeout_s`` is not from 0 to 30 days, the longest wait a
                member takes, or ``num_frames`` is outside 1 to 100,000.
                Nothing reached the camera.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, absent at the call or removed during the test. Only a
                capture failure is counted: any other raise from a grab ends
                the test as itself.
        """
        refuse_outside(num_frames, 'num_frames', _BANDWIDTH_FRAMES)
        refuse_unless_wait_s(timeout_s, 'timeout_s')
        self._scope.camera.refuse_camera_not_connected('run_camera_bandwidth_test')
        results = {
            'num_frames_requested': int(num_frames),
            'num_frames_received': 0,
            'num_frames_failed': 0,
            'failed_by_reason': {},
            'total_bytes': 0,
            'elapsed_seconds': 0,
            'mb_per_second': 0.0,
            'fps_actual': 0.0,
            'frame_sizes': [],
            'errors': [],
            'passed': True,
        }

        # What the frames were taken at, so a throughput reads against its
        # frame size and format: the values the camera holds, which cannot
        # fail, so a label never ends the test.
        frame_size = self._scope.camera.frame_size
        results['resolution'] = f'{frame_size.width}x{frame_size.height}' if frame_size else None
        results['pixel_format'] = self._scope.camera.pixel_format

        frame_size_set = set()
        start = time.monotonic()
        for i in range(int(num_frames)):
            if progress_cb and i % 250 == 0:
                try:
                    progress_cb(int(100 * i / max(num_frames, 1)), f'Frame {i}/{num_frames}')
                except Exception:
                    pass
            try:
                # force_to_8bit=False keeps native depth so frame size
                # reflects the actual bytes the SDK delivered.
                frame = self._scope.imaging.get_image(force_to_8bit=False, force_new_capture=True)
            except CaptureFailedError as failed:
                results['num_frames_failed'] += 1
                by_reason = results['failed_by_reason']
                by_reason[failed.reason] = by_reason.get(failed.reason, 0) + 1
                if len(results['errors']) < 20:
                    results['errors'].append(f'Frame {i}: {failed}')
            else:
                results['num_frames_received'] += 1
                nbytes = getattr(frame, 'nbytes', None) or len(frame)
                results['total_bytes'] += nbytes
                frame_size_set.add(int(nbytes))

            if time.monotonic() - start > timeout_s:
                results['errors'].append(f'Timeout at frame {i} after {timeout_s}s')
                results['passed'] = False
                break

        elapsed = time.monotonic() - start
        results['elapsed_seconds'] = round(elapsed, 2)
        if elapsed > 0:
            results['mb_per_second'] = round(results['total_bytes'] / (1024 * 1024) / elapsed, 2)
            results['fps_actual'] = round(results['num_frames_received'] / elapsed, 1)
        results['frame_sizes'] = sorted(frame_size_set)

        if results['num_frames_failed'] > 0:
            results['passed'] = False
        if len(frame_size_set) > 1:
            results['passed'] = False
            results['errors'].append(
                f'Inconsistent frame sizes: {sorted(frame_size_set)} -- '
                f'possible data corruption or config change during test'
            )

        logger.info(
            f'[SCOPE API ] run_camera_bandwidth_test: {results["num_frames_received"]}/{num_frames} '
            f'frames in {results["elapsed_seconds"]}s '
            f'({results["mb_per_second"]} MB/s, {results["fps_actual"]} fps), '
            f'passed={results["passed"]}'
        )
        return results

    @api(in_process=True, engineering=True)
    def run_grab_lifecycle_benchmark(
        self,
        num_cycles: int = 100,
        inter_cycle_delay_s: float = 0.0,
        vary_settings: bool = False,
        *,
        slow_threshold_s: float = 3.0,
        progress_cb: ProgressCallback | None = None,
    ) -> dict:
        """Characterize stop_grabbing/start_grabbing latency, on the camera lane, and wait.

        Each cycle stops the camera's grab and starts it again, timing both:
        a measured floor for the interval a camera needs between a stop and
        the next start. The typical cycle is 130-150 ms; one near 11 s has
        been seen when a stop lands within about 275 ms of a start, before
        the camera produced a frame. Sweeping ``inter_cycle_delay_s`` across
        runs shows the smallest delay with no slow cycle. The benchmark is
        one command on the camera's lane, waited for, bounded per cycle.

        Args:
            num_cycles: Stop-and-start cycles to run, 1 to 10,000.
            inter_cycle_delay_s: Seconds between a stop and the next start
                (and any settings change), 0 to 60.
            vary_settings: True to alternate the gain (1 and 4 dB) and the
                exposure (10 and 50 ms) between cycles, as a protocol's steps
                do.
            slow_threshold_s: A cycle's seconds at which it counts as slow,
                0 to 600; slow cycles are counted apart.
            progress_cb: Called as ``callback(percent, message)`` every 10
                cycles, on the camera lane's worker.

        Returns:
            dict: ``num_cycles``, ``inter_cycle_delay_s``,
                ``vary_settings``, ``slow_threshold_s``,
                ``slow_cycle_count``, ``slow_cycles`` (each ``idx``,
                ``cycle_s``, ``stop_s``, ``start_s``), the 50th, 95th and
                99th percentiles of the cycle, stop and start times
                (``cycle_p50`` ...), ``total_elapsed_s``, ``camera_model``,
                ``pylon_version``, ``errors`` and ``written_to``.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or once a cycle finds it removed.
            ArgumentRefusedError: ``'not_a_number'``, a time is not a finite
                number; ``'out_of_range'``, ``num_cycles`` is outside 1 to
                10,000, ``inter_cycle_delay_s`` outside 0 to 60 or
                ``slow_threshold_s`` outside 0 to 600. Nothing reached the
                camera.
        """
        for name, value, limits in (
            ('num_cycles', num_cycles, _BENCHMARK_CYCLES),
            ('inter_cycle_delay_s', inter_cycle_delay_s, _BENCHMARK_DELAY_S),
            ('slow_threshold_s', slow_threshold_s, _BENCHMARK_SLOW_S),
        ):
            refuse_outside(value, name, limits)
        bound_s = num_cycles * (self._GRAB_CYCLE_BOUND_S + inter_cycle_delay_s)
        return self._scope.camera._dispatch_camera(
            self._run_grab_lifecycle_benchmark_impl,
            'run_grab_lifecycle_benchmark',
            args=(num_cycles, inter_cycle_delay_s, vary_settings),
            kwargs={'slow_threshold_s': slow_threshold_s, 'progress_cb': progress_cb},
            timeout_s=max(bound_s, self._GRAB_CYCLE_BOUND_S),
        )

    def _run_grab_lifecycle_benchmark_impl(
        self,
        num_cycles: int = 100,
        inter_cycle_delay_s: float = 0.0,
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
        produces a frame. Sweeping ``inter_cycle_delay_s`` through
        0/0.05/0.1/0.2/0.5/1 s across runs reveals the smallest delay
        that yields ZERO slow cycles.

        Stays inside the API: drops to ``self._scope._camera_driver.stop_grabbing`` /
        ``start_grabbing`` directly, which is a Rule-1 downward call from
        the API into its driver -- same pattern as ``set_frame_size`` etc.

        Args:
            num_cycles: Stop/start cycles to perform.
            inter_cycle_delay_s: Sleep between StopGrabbing and the next
                StartGrabbing (and any settings churn), in seconds.
            vary_settings: When True, alternate gain (1.0 <-> 4.0) and
                exposure (10 ms <-> 50 ms) between cycles to reproduce the
                per-step protocol pattern that caused STALL-1.
            slow_threshold_s: Cycle wall-time considered "slow" -- counted
                separately so the operator sees how often the pathological
                case fires under the chosen delay.
            progress_cb: Optional ``callback(percent_int, message_str)``
                called every 10 cycles.

        Returns:
            dict with: num_cycles, inter_cycle_delay_s, vary_settings,
                slow_threshold_s, slow_cycle_count, slow_cycles (list of
                {idx, cycle_s, stop_s, start_s}), cycle_p50/p95/p99,
                stop_p50/p95/p99, start_p50/p95/p99, total_elapsed_s,
                camera_model, pylon_version, errors, written_to.
        """
        results = {
            'num_cycles': int(num_cycles),
            'inter_cycle_delay_s': float(inter_cycle_delay_s),
            'vary_settings': bool(vary_settings),
            'slow_threshold_s': float(slow_threshold_s),
            'slow_cycle_count': 0,
            'slow_cycles': [],
            'cycle_p50_s': 0.0,
            'cycle_p95_s': 0.0,
            'cycle_p99_s': 0.0,
            'stop_p50_s': 0.0,
            'stop_p95_s': 0.0,
            'stop_p99_s': 0.0,
            'start_p50_s': 0.0,
            'start_p95_s': 0.0,
            'start_p99_s': 0.0,
            'total_elapsed_s': 0.0,
            'camera_model': None,
            'pylon_version': None,
            'errors': [],
            'written_to': None,
        }

        if not self._scope._camera_driver or not self._scope._camera_driver.active:
            results['errors'].append('Camera not active')
            return results

        results['camera_model'] = self._scope.capabilities.camera_model
        results['pylon_version'] = self._scope.capabilities.camera_sdk_version

        cycle_times, stop_times, start_times = [], [], []
        delay_s = max(0.0, float(inter_cycle_delay_s))

        # Snapshot current settings so we can restore even when vary_settings
        # is on -- the benchmark must not leave the camera in an arbitrary state.
        saved_camera_state = (
            self._scope.camera.save_camera_state('grab_lifecycle_benchmark')
            if vary_settings
            else None
        )

        t_overall_start = time.monotonic()
        for i in range(int(num_cycles)):
            if progress_cb and i % 10 == 0:
                try:
                    progress_cb(int(100 * i / max(num_cycles, 1)), f'Cycle {i}/{num_cycles}')
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
                        self._scope.camera.set_gain_db(1.0)
                        self._scope.camera.set_exposure_ms(10.0)
                    else:
                        self._scope.camera.set_gain_db(4.0)
                        self._scope.camera.set_exposure_ms(50.0)

                t1 = time.monotonic()
                self._scope._camera_driver.start_grabbing()
                start_s = time.monotonic() - t1
            except Exception as e:
                # A camera gone ends the benchmark; the lane answers for it.
                if self._scope.camera.camera_removed:
                    raise
                results['errors'].append(f'Cycle {i}: {type(e).__name__}: {e}')
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
                    results['slow_cycles'].append(
                        {
                            'idx': i,
                            'cycle_s': round(cycle_s, 4),
                            'stop_s': round(stop_s, 4),
                            'start_s': round(start_s, 4),
                        }
                    )

        results['total_elapsed_s'] = round(time.monotonic() - t_overall_start, 3)

        # Restore the caller's camera settings so vary_settings doesn't leak state.
        try:
            if saved_camera_state is not None:
                self._scope.camera.restore_camera_state(saved_camera_state)
        except Exception as e:
            results['errors'].append(f'Restore settings failed: {type(e).__name__}: {e}')

        def _pct(samples, q):
            if not samples:
                return 0.0
            return round(float(np.percentile(samples, q)), 4)

        results['cycle_p50_s'] = _pct(cycle_times, 50)
        results['cycle_p95_s'] = _pct(cycle_times, 95)
        results['cycle_p99_s'] = _pct(cycle_times, 99)
        results['stop_p50_s'] = _pct(stop_times, 50)
        results['stop_p95_s'] = _pct(stop_times, 95)
        results['stop_p99_s'] = _pct(stop_times, 99)
        results['start_p50_s'] = _pct(start_times, 50)
        results['start_p95_s'] = _pct(start_times, 95)
        results['start_p99_s'] = _pct(start_times, 99)

        # Persist under the log folder (camera_timing/) keyed by model + sdk
        # version + delay so a sweep across delays produces one file per data
        # point. The log folder is the support-bundle root, so a benchmark run
        # travels with the logs; writing under the source tree instead both
        # polluted the checkout and left the data point out of the bundle.
        # Two runs with the same model, SDK and delay inside one second used
        # to share a name and the later one overwrote the earlier: the stamp
        # carries microseconds, and the file is created exclusively so a name
        # that is somehow taken is an error here, never an overwrite.
        try:
            import json

            model = results['camera_model'] or 'unknown_camera'
            sdk = results['pylon_version'] or 'unknown_sdk'
            safe_model = str(model).replace(' ', '_').replace('/', '_')
            safe_sdk = str(sdk).replace(' ', '_').replace('/', '_')
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            timing_dir = pathlib.Path(log_dir) / 'camera_timing'
            timing_dir.mkdir(parents=True, exist_ok=True)
            out_path = timing_dir / (
                f'grab_lifecycle_benchmark_{safe_model}_sdk{safe_sdk}_'
                f'delay{inter_cycle_delay_s:g}s_{ts}.json'
            )
            with open(out_path, 'x') as f:
                json.dump(results, f, indent=2)
            results['written_to'] = str(out_path)
        except Exception as e:
            results['errors'].append(f'Persist failed: {type(e).__name__}: {e}')

        logger.info(
            f'[SCOPE API ] run_grab_lifecycle_benchmark: {num_cycles} cycles, '
            f'delay={inter_cycle_delay_s}s, vary={vary_settings} -> '
            f'cycle p50={results["cycle_p50_s"]}s p95={results["cycle_p95_s"]}s '
            f'p99={results["cycle_p99_s"]}s, slow={results["slow_cycle_count"]} '
            f'(>={slow_threshold_s}s), total={results["total_elapsed_s"]}s'
        )
        return results

    @api(in_process=True, engineering=True)
    def run_pylon_diagnostic_probe(
        self,
        duration_s: float = 3.0,
        *,
        drain_camera_side_errors: bool = True,
        progress_cb: ProgressCallback | None = None,
    ) -> dict:
        """One-shot Pylon-camera diagnostic probe, on the camera lane, and wait.

        Records the camera's identity, its configuration and the change in
        its stream statistics over ``duration_s``, with the host's OS, name
        and camera SDK version, in one JSON file under the log folder's
        ``camera_probe/``, named for the model, serial, firmware, host and
        time so a sweep's files sort together. The grab state is not
        changed: a camera not grabbing shows counters near zero, so call it
        with the live view running. The probe is one command on the camera's
        lane, waited for. It pops the camera's error queue, so it is a write
        to the camera and refused while another activity holds the scope.

        Args:
            duration_s: The sampling window in seconds, 0 to 3,600.
            drain_camera_side_errors: True to drain the camera's own error
                queue and record its codes, which Basler documents for its
                support only.
            progress_cb: Called as ``callback(percent, message)`` at the
                start, mid-window and the end, on the camera lane's worker.

        Returns:
            dict: The camera's snapshot with the host, the times and the
                ``output_path``; ``{'connected': False, 'errors': [...]}``
                with no camera active, and ``'supported': False`` besides for
                a camera whose driver has no probe.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected.
            ArgumentRefusedError: ``'not_a_number'``, ``duration_s`` is not a finite
                number; ``'out_of_range'``, it is outside 0 to 3,600.
                Nothing reached the camera.
        """
        refuse_outside(duration_s, 'duration_s', _PROBE_DURATION_S)
        return self._scope.camera._dispatch_camera(
            self._run_pylon_diagnostic_probe_impl,
            'run_pylon_diagnostic_probe',
            args=(duration_s,),
            kwargs={
                'drain_camera_side_errors': drain_camera_side_errors,
                'progress_cb': progress_cb,
            },
            timeout_s=duration_s + self._GRAB_CYCLE_BOUND_S,
        )

    def _run_pylon_diagnostic_probe_impl(
        self,
        duration_s: float = 3.0,
        *,
        drain_camera_side_errors: bool = True,
        progress_cb=None,
    ) -> dict:
        """One-shot Pylon-camera diagnostic probe with JSON output.

        Captures camera identity, current configuration, and stream-
        grabber statistics counter deltas over a sampling window of
        ``duration_s`` seconds. Adds host metadata (OS, hostname, and the
        active camera SDK name + version) and writes a single JSON file
        to ``<log folder>/camera_probe/`` keyed on
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
                camera is active, and ``{'connected': False, 'supported':
                False, 'errors': [...]}`` for a camera whose driver has no
                probe.
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

        if progress_cb is not None:
            try:
                progress_cb(70, 'snapshot captured; collecting host metadata')
            except Exception:
                pass

        # Host metadata
        import socket
        import platform as _platform

        # Driver-neutral SDK provenance: stamp whichever camera SDK actually
        # produced this snapshot (Basler pylon, IDS peak, ...), not a hardcoded
        # Pylon assumption. Every Camera driver implements get_sdk_info (base
        # class default for SDK-less drivers).
        snapshot['host'] = {
            'os': self._human_os_version(),
            'hostname': socket.gethostname(),
            'machine': _platform.machine(),
            'camera_sdk': self._scope._camera_driver.get_sdk_info(),
        }

        now_utc = datetime.datetime.now(datetime.UTC)
        end_iso = now_utc.isoformat()
        start_iso = (
            now_utc - datetime.timedelta(seconds=snapshot.get('duration_s_actual', duration_s))
        ).isoformat()
        snapshot['timestamps'] = {'start_iso': start_iso, 'end_iso': end_iso}

        # Filter-by-load top-level keys (per v4 author request: easier
        # to grep across many files than parsing camera.firmware_version
        # nested)
        snapshot['firmware_version'] = snapshot.get('camera', {}).get('firmware_version')

        dltl_token = self._dltl_filename_token(snapshot.get('config', {}))
        snapshot['dltl_config'] = dltl_token

        # JSON file write -- under the log folder so the snapshot travels with
        # the support bundle instead of landing in the source tree.
        try:
            import json

            out_dir = pathlib.Path(log_dir) / 'camera_probe'
            out_dir.mkdir(parents=True, exist_ok=True)

            def _safe_token(v: str | None, fallback: str) -> str:
                s = str(v) if v is not None else fallback
                # Filenames: replace separators that would break the
                # __ split-pattern, and any path separators.
                for bad in (' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|'):
                    s = s.replace(bad, '_')
                return s

            model_t = _safe_token(snapshot.get('camera', {}).get('model_name'), 'unknown_model')
            serial_t = _safe_token(snapshot.get('camera', {}).get('serial'), 'unknown_serial')
            fw_t = _safe_token(snapshot.get('firmware_version'), 'unknown_fw')
            host_t = _safe_token(snapshot['host']['hostname'], 'unknown_host').replace('.', '_')
            # Microseconds, and an exclusive create: two probes of one camera
            # inside one second would otherwise share a name, and the later
            # would overwrite the earlier.
            ts_t = now_utc.strftime('%Y%m%dT%H%M%S%fZ')

            fname = f'{model_t}__sn{serial_t}__fw{fw_t}__{host_t}__{dltl_token}__{ts_t}.json'
            out_path = out_dir / fname
            with open(out_path, 'x') as f:
                json.dump(snapshot, f, indent=2, default=str)
            snapshot['output_path'] = str(out_path)
        except Exception as e:
            snapshot.setdefault('errors', []).append(f'JSON write failed: {type(e).__name__}: {e}')

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
        timeout_s: float | None = None,
    ) -> str:
        """Send a single firmware diagnostic command and return the response.

        Wraps the driver's ``exchange_command`` with API-layer logging.
        Diagnostic clients (tech-support report, bench tools) MUST go
        through this method instead of reaching the driver directly
        (LV-24 / LV-32 / LV-40).

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string (e.g. ``'INFO'``, ``'FACTORY'``).
            response_numlines: Forwarded to driver; how many response lines
                to read before returning (driver-specific default if None).
            timeout_s: Per-call serial timeout in seconds, or None for the
                driver's default.

        Returns:
            str: Response from the board, ``'Board not connected'`` with no
                board connected (``_board_connected``), or ``'Error: <msg>'``
                if the exchange raised.

        Raises:
            ValueError: A motor command that moves, stops or repositions a
                motor; the motion API carries those.
            HardwareCommandRefusedError: a run, a diagnostic or a home holds the
                scope and this call is not made under its taking; the
                command is not sent. An exchange that fails once sent is
                still answered with ``'Error: <msg>'``.
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                ``timeout_s`` is not a wait from 0 to 30 days. Nothing was
                sent.
        """
        if isinstance(target, str) and target.lower() in ('motor', 'motion'):
            _refuse_motor_verb(command)
        if timeout_s is not None:
            refuse_unless_wait_s(timeout_s, 'timeout_s')
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command: {e}')
            return f'{ERROR_PREFIX}{e}'

        if not self._board_connected(board):
            return NOT_CONNECTED

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command(target={target}, command={command!r}, '
            f'response_numlines={response_numlines}, timeout_s={timeout_s})'
        )
        kwargs = {}
        if response_numlines is not None:
            kwargs['response_numlines'] = response_numlines
        if timeout_s is not None:
            # driver exchange_command keeps bare `timeout` (pyserial-shaped)
            kwargs['timeout'] = timeout_s
        return self._scope.motion._dispatch_motion(
            self._exchange_command_impl,
            'send_diagnostic_command',
            args=(board, target, command),
            kwargs=kwargs,
            timeout_s=(timeout_s or 0.0) + self._BOARD_QUEUE_MARGIN_S,
            slow_task_threshold_sec=timeout_s,
        )

    @staticmethod
    def _exchange_command_impl(board, target: str, command: str, **kwargs) -> str:
        try:
            resp = board.exchange_command(command, **kwargs)
            return resp if resp is not None else NO_REPLY
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command({target}, {command!r}) failed: {e}'
            )
            return f'{ERROR_PREFIX}{e}'

    def send_diagnostic_command_multiline(
        self,
        target: str,
        command: str,
        *,
        timeout_s: float = 60,
        end_markers: list[str] | None = None,
    ) -> str | list[str]:
        """Send a firmware diagnostic command expected to return multiple lines.

        For SELFTEST, INFO with multi-line output, etc. Wraps the driver's
        ``exchange_multiline`` with API-layer logging.

        Args:
            target: 'led' or 'motor'.
            command: Firmware command string.
            timeout_s: Total timeout in seconds.
            end_markers: Substrings marking end-of-response. Default
                ``['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']``.

        Returns:
            Response (driver-defined; typically str or list[str]),
            ``'Board not connected'``, or ``'Error: <msg>'``.

        Raises:
            ValueError: A motor command that moves, stops or repositions a
                motor; the motion API carries those.
            HardwareCommandRefusedError: a run, a diagnostic or a home holds the
                scope and this call is not made under its taking; the
                command is not sent. An exchange that fails once sent is
                still answered with ``'Error: <msg>'``.
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                ``timeout_s`` is not a wait from 0 to 30 days. Nothing was
                sent.
        """
        if isinstance(target, str) and target.lower() in ('motor', 'motion'):
            _refuse_motor_verb(command)
        refuse_unless_wait_s(timeout_s, 'timeout_s')
        try:
            board = self._diagnostic_target_board(target)
        except ValueError as e:
            logger.warning(f'[SCOPE API ] send_diagnostic_command_multiline: {e}')
            return f'{ERROR_PREFIX}{e}'

        if not self._board_connected(board):
            return NOT_CONNECTED

        if end_markers is None:
            end_markers = ['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']

        logger.debug(
            f'[SCOPE API ] send_diagnostic_command_multiline(target={target}, '
            f'command={command!r}, timeout_s={timeout_s}, end_markers={end_markers})'
        )
        return self._scope.motion._dispatch_motion(
            self._exchange_multiline_impl,
            'send_diagnostic_command_multiline',
            args=(board, target, command, timeout_s, end_markers),
            timeout_s=timeout_s + self._BOARD_QUEUE_MARGIN_S,
            slow_task_threshold_sec=timeout_s,
        )

    @staticmethod
    def _exchange_multiline_impl(board, target, command, timeout_s, end_markers):
        try:
            # driver exchange_multiline keeps bare `timeout` (pyserial-shaped)
            result = board.exchange_multiline(command, timeout=timeout_s, end_markers=end_markers)
            return result if result else NO_RESPONSE
        except Exception as e:
            logger.warning(
                f'[SCOPE API ] send_diagnostic_command_multiline({target}, {command!r}) failed: {e}'
            )
            return f'{ERROR_PREFIX}{e}'

    # --- Motor driver / fan diagnostics ---
    # Each read returns parsed values or None when the firmware does not
    # support the command (legacy 2024-09-10 firmware did not include
    # DRVSTAT_<axis> / FANSPEED / FAN). Per Eric: the driver
    # owns firmware-version gating; callers (TSR, future REST
    # diagnostic endpoint) read None as "INCONCLUSIVE -- firmware
    # does not support this probe." The fan setter differs on purpose: a
    # write that did not happen raises, so a caller cannot read a status
    # for one that did.

    @api(in_process=True, reads=True, engineering=True)
    def read_motor_drv_status(self, axis: Axis) -> int | None:
        """Read TMC5072 DRV_STATUS register for an axis.

        Returns the raw register value as int (caller decodes bits),
        or None when the firmware does not implement DRVSTAT_<axis>.

        Args:
            axis: The axis whose driver chip to read.

        Raises:
            ArgumentRefusedError: ``'not_offered'``, ``axis`` is no axis
                name; nothing was read.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_drv_status'):
            return None
        return drv.read_drv_status(axis)

    @api(in_process=True, reads=True, engineering=True)
    def read_motor_fan_rpm(self) -> int | None:
        """Read motor-board fan tachometer RPM.

        Returns RPM as int (0 if no tach wire) or None when firmware
        does not implement FANSPEED.

        Raises:
            Nothing.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'read_fanspeed'):
            return None
        return drv.read_fanspeed()

    @api(in_process=True, reads=True, engineering=True)
    def motor_fan_spin_times(self) -> dict[str, float] | None:
        """How long the motor-board fan takes to reach a new duty, in seconds.

        ``{'spin_up_s': ..., 'spin_down_s': ...}``: up to speed from a stop,
        and down to a stop from speed, as the board's driver states them. A
        ``read_motor_fan_rpm`` sooner after ``set_motor_fan_duty`` reads the
        old speed. None when no motor board's driver drives a fan; whether
        the board's firmware takes a fan write is ``set_motor_fan_duty``'s
        refusal, asked on the board.

        Raises:
            Nothing.
        """
        drv = getattr(self._scope, '_motion_driver', None)
        if drv is None or not hasattr(drv, 'set_fan_duty'):
            return None
        return {'spin_up_s': drv.fan_spin_up_s, 'spin_down_s': drv.fan_spin_down_s}

    @api(in_process=True, engineering=True)
    def set_motor_fan_duty(self, duty_pct: int) -> None:
        """Set motor-board fan PWM duty cycle (0..100), and wait for the board to take it.

        The fan reaches the new duty later: ``motor_fan_spin_times``.

        Args:
            duty_pct: The fan's PWM duty cycle, in percent, 0 to 100.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` (the motor
                controller); ``'axis_absent'``, ``MissingPart.MOTORS`` on a
                manual scope; or ``'not_on_this_scope'``, ``MissingPart.FAN_CONTROL``
                for a controller without fan control. Nothing was sent.
            ArgumentRefusedError: ``'fan_duty_out_of_range'``, ``duty_pct``
                is outside 0..100. Asked after the hardware, before anything
                is sent.
            BoardNoReplyError: ``'board_no_reply'``, the board did not answer
                the write.
            BoardReportedError: ``'board_reported_error'``, the board answered
                the write with an error.
        """
        self._scope.motion.refuse_absent('set_motor_fan_duty')
        if not 0 <= duty_pct <= 100:
            raise ArgumentRefusedError('fan_duty_out_of_range', argument='duty_pct', value=duty_pct)
        self._scope.motion._dispatch_motion(
            self._set_motor_fan_duty_impl,
            'set_motor_fan_duty',
            args=(self._scope._motion_driver, duty_pct),
            timeout_s=self._BOARD_QUEUE_MARGIN_S,
        )

    @staticmethod
    def _set_motor_fan_duty_impl(drv, duty_pct: int) -> None:
        # Asked on the lane: the support probe is a board exchange.
        if not drv.supports_fan():
            part = MissingPart.FAN_CONTROL
            raise HardwareCommandRefusedError(part.reason, 'set_motor_fan_duty', missing=part)
        drv.set_fan_duty(duty_pct)

    # --- LED engineering mode (LEDREADS / SELFTEST handshake) ---
    # Open-coded FACTORY / Y / Q sequences in callers were leaving the
    # LED board wedged when LEDREADS or SELFTEST timed out mid-eng-mode
    # -- Q was sent without the post-Q drain + end-marker check that the
    # driver-canonical methods do, so the firmware was left in a state
    # where every subsequent LED command returned ''. These sub-API
    # entries keep the careful handshake as the single canonical
    # implementation.

    def _refuse_no_led_engineering_mode(self, member: str) -> None:
        """Refuse when no LED controller is connected, or it has no engineering mode.

        No LED capability names the handshake, so the driver's having the
        method is the dispatch: the Classic FX2 LED controller has none.
        """
        self._scope.illumination.refuse_controller_not_connected(member)
        if not hasattr(self._scope._led_driver, 'enter_engineering_mode'):
            part = MissingPart.LED_ENGINEERING_MODE
            raise HardwareCommandRefusedError(part.reason, member, missing=part)

    @api(in_process=True, engineering=True)
    def enter_led_engineering_mode(self, timeout_s: float = 5.0) -> None:
        """Enter LED engineering mode via the driver-canonical handshake, and wait for it.

        Args:
            timeout_s: Max seconds to wait for the engineering-mode
                handshake to complete.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` (the LED
                controller) or ``'not_on_this_scope'``
                (``MissingPart.LED_ENGINEERING_MODE``, a controller without
                the handshake). Nothing was sent.
            BoardNoReplyError: ``'board_no_reply'``, the board did not answer
                FACTORY.
            BoardReportedError: ``'board_reported_error'``, the board did not
                complete the handshake: no Y/N prompt; or no engineering-mode
                banner after Y, when the driver has already brought it back
                to safe mode.
            LedFirmwareWedgedError: ``'led_firmware_wedged'``, the handshake
                failed after Y and the board's soft reset did not bring its
                firmware back; it needs a power cycle.
            ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``,
                ``timeout_s`` is not a wait from 0 to 30 days. Nothing was
                sent.
        """
        self._refuse_no_led_engineering_mode('enter_led_engineering_mode')
        refuse_unless_wait_s(timeout_s, 'timeout_s')
        self._scope.motion._dispatch_motion(
            self._enter_led_engineering_mode_impl,
            'enter_led_engineering_mode',
            args=(self._scope._led_driver, timeout_s),
            timeout_s=timeout_s + self._BOARD_QUEUE_MARGIN_S,
        )

    @staticmethod
    def _enter_led_engineering_mode_impl(drv, timeout_s: float) -> None:
        drv.enter_engineering_mode(timeout=timeout_s)

    @api(in_process=True, engineering=True)
    def exit_led_engineering_mode(self) -> None:
        """Exit LED engineering mode via the driver-canonical handshake, and wait for it.

        Driver method drains and sleeps after Q so the LED firmware
        actually transitions out of eng mode.

        Raises:
            HardwareCommandRefusedError: as ``enter_led_engineering_mode``.
            LedFirmwareWedgedError: ``'led_firmware_wedged'``, Q failed and
                the board's soft reset did not bring its firmware back; it
                needs a power cycle.
        """
        self._refuse_no_led_engineering_mode('exit_led_engineering_mode')
        self._scope.motion._dispatch_motion(
            self._exit_led_engineering_mode_impl,
            'exit_led_engineering_mode',
            args=(self._scope._led_driver,),
            timeout_s=self._BOARD_QUEUE_MARGIN_S,
        )

    @staticmethod
    def _exit_led_engineering_mode_impl(drv) -> None:
        drv.exit_engineering_mode()

    # --- Board identity ---

    @api(reads=True)
    def get_motor_info(self) -> MotorBoardInfo:
        """The motor board's identity.

        Served from what FULLINFO and INFO cached at connect, never re-queried
        over the serial bus: identity is fixed for the life of a connection,
        and this is read for every saved frame's metadata.

        Returns:
            MotorBoardInfo: Each field None when unknown, all of them with
                no board connected -- none installed, or one lost since
                connect, whose cached identity no longer describes a board
                that is there.

        Raises:
            Nothing.
        """
        if not self._scope.motor_connected:
            return MotorBoardInfo(
                model=None,
                serial_number=None,
                firmware_version=None,
                firmware_date=None,
                command_set=None,
            )
        driver = self._scope._motion_driver
        speaks_text = hasattr(driver, 'exchange_multiline')
        return MotorBoardInfo(
            model=driver.get_microscope_model(),
            serial_number=driver.get_serial_number(),
            **_firmware_identity(driver),
            command_set=MotorCommandSet.TEXT if speaks_text else None,
        )

    @api(reads=True)
    def get_led_info(self) -> LedBoardInfo:
        """The LED board's firmware and command set.

        Returns:
            LedBoardInfo: Each field None when unknown, all of them with no
                board connected.

        Raises:
            Nothing.
        """
        drv = self._scope._led_driver
        if not drv or not drv.is_connected():
            return LedBoardInfo(firmware_version=None, firmware_date=None, command_set=None)

        if not hasattr(drv, 'exchange_multiline'):
            command_set = None
        elif drv.is_v2:
            command_set = LedCommandSet.V2
        else:
            command_set = LedCommandSet.LEGACY
        return LedBoardInfo(**_firmware_identity(drv), command_set=command_set)

    @api(in_process=True, reads=True, engineering=True)
    def read_led_currents_ma(self) -> dict[int, float | None]:
        """Read the measured current of every LED channel, in mA.

        One pass over the board's channels on the IO lane. The v2 firmware
        answers these reads only in engineering mode; the caller enters it.

        Returns:
            dict: Channel -> mA, or None for a channel whose current could
                not be read (no reply, an unparseable one, or a board with
                no current sensing). Empty when no LED board is connected.

        Raises:
            Nothing.
        """
        drv = self._scope._led_driver
        if not self._scope.led_connected:
            return {}
        return self._scope.motion._dispatch_motion(
            self._read_led_currents_impl,
            'read_led_currents_ma',
            args=(drv,),
            timeout_s=self._BOARD_QUEUE_MARGIN_S,
        )

    @staticmethod
    def _read_led_currents_impl(drv) -> dict[int, float | None]:
        return {ch: drv.read_led_current(ch) for ch in drv.available_channels()}

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
            return f'dltl{round(value / 1_000_000)}M'
        return 'dltlunknown'

    @staticmethod
    def _board_connected(board) -> bool:
        """Whether a diagnostic may reach ``board``: found at bring-up, and not lost since.

        A board lost mid-session stays lost until LumaViewPro restarts, as
        every other command treats it; an exchange sent to it would reach
        the serial layer's reopen and bring it back under the API, its axes
        answering positions held from before the loss.
        """
        return board is not None and getattr(board, 'found', False) and board.is_connected()

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
            f"send_diagnostic_command: unknown target {target!r} (expected 'led' or 'motor')"
        )
