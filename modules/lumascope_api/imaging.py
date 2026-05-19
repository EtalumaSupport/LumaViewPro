# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI -- sub-API for camera capture / image acquisition.

Wave 7 Phase 4c -- stateless bodies relocated from Lumascope. The
remaining ~30 stateful bodies + state slots (_camera_cache, _frame_buffer,
_scale_bar, _capturing_event, _focusing_event, _camera_listeners,
frame_validity instance) relocate in Phase 4d. Lumascope keeps Rule-30
one-line forwarders until Phase 4f.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.3 for the canonical
method list. Frame-listener registry and live_processing plugin
infrastructure ship in Wave 7 Phase 4d.5.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING

from lvp_logger import logger

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.camera import Camera

_api_log = _logging.getLogger('LVP.api')


class ImagingAPI:
    """Imaging sub-API. Owns stateless camera setters/getters/passthroughs
    after Phase 4c; full body + state ownership in Phase 4d/4f.
    """

    def __init__(self, scope: 'Lumascope', driver: 'Camera | None') -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._camera_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding ImagingAPI. Same pattern as MotionAPI._driver /
        # IlluminationAPI._driver (Wave 7 Phase 2b / 3c precedent).
        del driver  # noqa -- intentionally unused, kept for backward call sites

    @property
    def _driver(self) -> 'Camera | None':
        """Resolve the camera driver via the composition root each access.

        Lumascope's `_camera_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps ImagingAPI in sync without rebinding.
        """
        return self._scope._camera_driver

    # --- Setters ---
    def set_gain(self, *args, **kwargs):
        return self._scope.set_gain(*args, **kwargs)

    def set_exposure_time(self, *args, **kwargs):
        return self._scope.set_exposure_time(*args, **kwargs)

    def set_auto_gain(self, *args, **kwargs):
        return self._scope.set_auto_gain(*args, **kwargs)

    def set_auto_exposure_time(self, *args, **kwargs):
        return self._scope.set_auto_exposure_time(*args, **kwargs)

    def set_frame_size(self, *args, **kwargs):
        return self._scope.set_frame_size(*args, **kwargs)

    def set_binning_size(self, *args, **kwargs):
        return self._scope.set_binning_size(*args, **kwargs)

    def set_pixel_format(self, *args, **kwargs):
        return self._scope.set_pixel_format(*args, **kwargs)

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        """Set BslAcquisitionStopMode (Pylon-only; no-op on IDS).

        Controls camera behavior when StopGrabbing fires during an
        in-flight exposure:

          - ``'Complete'`` (Pylon default): waits for the current
            exposure to finish before stopping.
          - ``'CancelExposure'``: stops cleanly; partial frame
            discarded.
          - ``'AbortExposure'``: aborts immediately; partial frame
            discarded.

        Default ``'Complete'`` waits up to the full exposure on long
        fluorescence captures (5-10 s) -- presents identically to a
        multi-second app-side stall when the user toggles modes.
        ``'AbortExposure'`` resolves the symptom but is bench-
        unvalidated on Etaluma's cameras. Setter is provided for
        bench characterization; default is unchanged.

        Args:
            mode: One of ``'Complete'``, ``'CancelExposure'``,
                ``'AbortExposure'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, mode is invalid, the driver doesn't
                implement the setter (IDS), or
                BslAcquisitionStopMode is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_acquisition_stop_mode'):
            logger.warning(
                f'[SCOPE API ] set_acquisition_stop_mode: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        try:
            return bool(self._driver.set_acquisition_stop_mode(mode=mode))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting acquisition_stop_mode: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "BslAcquisitionStopMode change failed",
                f"Could not set acquisition_stop_mode to {mode!r}: "
                f"{type(ex).__name__}: {ex}. Camera may still be at "
                f"the previous stop-mode setting."
            )
            raise

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """Set BandwidthReserveMode (GigE-only Pylon node).

        ``'Default'`` reserves a portion of GigE bandwidth for
        retransmits; ``'Performance'`` dedicates all bandwidth to
        image transmit. Per dmA3536-9gm spec, ``'Performance'``
        unlocks 9.5 fps vs the default 9.3 fps.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            mode: ``'Default'`` or ``'Performance'``.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter,
                or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_bandwidth_reserve_mode'):
            return False
        try:
            return bool(self._driver.set_bandwidth_reserve_mode(mode=mode))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting BandwidthReserveMode: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "BandwidthReserveMode change failed",
                f"Could not set BandwidthReserveMode to {mode!r}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> bool:
        """Set the camera's DeviceLinkThroughputLimit mode and value.

        Both nodes are live-writable per the SDK lock-state table -- no
        StopGrabbing/StartGrabbing wrap. Per-camera defaults bench-
        witnessed (USB3): ace 2 a2A3536-31umBAS at 360 MB/s -> 28.8 fps;
        dart daA3840-45um at 160 MB/s -> 18.7 fps. Setting ``mode='Off'``
        lets the camera run at sensor-readout maximum (~31.2 fps ace 2;
        ~44.9 fps dart on USB3).

        Used by the diagnostic-probe sweep in ``tools/`` to characterize
        failure rate vs throughput across camera + firmware + host
        cells. Per Basler docs: "Corrupt or dropped frames may occur if
        the DeviceLinkThroughputLimit parameter is too high" -- bench-
        test failure rate alongside fps before settling on a per-camera
        production default.

        **Transport caveat (GigE):** on GigE cameras (e.g. dmA3536-9gm)
        DLTL is bounded above by the GigE wire limit (~110 MB/s usable
        on 1 Gbps Ethernet). Setting above wire limit is a no-op; below
        caps fps proportionally. For GigE bandwidth control use
        ``set_gev_inter_packet_delay`` / ``set_bandwidth_reserve_mode``
        instead -- those are the GigE-side tools.

        Args:
            mode: ``'On'`` or ``'Off'`` (case-sensitive; matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the mode argument is invalid, or the driver
                returned False (unsupported by this driver).

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_device_link_throughput_limit'):
            logger.warning(
                f'[SCOPE API ] set_device_link_throughput_limit: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return False
        try:
            return bool(self._driver.set_device_link_throughput_limit(
                mode=mode, value_bps=value_bps,
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting DLTL: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "DeviceLinkThroughputLimit change failed",
                f"Could not set DLTL to mode={mode}, value_bps={value_bps}: "
                f"{type(ex).__name__}: {ex}. Camera may still be at the "
                f"previous DLTL setting."
            )
            raise

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set Pylon StreamGrabber MaxTransferSize (USB3 only).

        Bytes-per-USB-transfer the SDK requests from the kernel. Per
        Basler `stream-grabber-parameters.html` this is the named lever
        for the symptom "fails to receive image stream" -- decreasing
        the value works around kernel / driver USB-transfer-size
        constraints on some Windows hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value_bytes: New MaxTransferSize in bytes.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_max_transfer_size'):
            return False
        try:
            return bool(self._driver.set_max_transfer_size(
                value_bytes=value_bytes
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting MaxTransferSize: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "MaxTransferSize change failed",
                f"Could not set MaxTransferSize to {value_bytes}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """Set Pylon StreamGrabber NumMaxQueuedUrbs (USB3 only).

        Number of USB Request Blocks the SDK keeps in flight to the
        kernel. Per Basler `stream-grabber-parameters.html` this is
        the named lever for "insufficient system memory"
        (0xe2010130 / 0xe2100001) -- decreasing the value reduces
        kernel URB allocation pressure on memory-constrained hosts.

        USB3-only. The node is absent on GigE cameras and on the IDS
        SDK; returns False so the bench-probe sweep can call this
        method unconditionally per cell.

        Args:
            value: New NumMaxQueuedUrbs (count).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, the driver doesn't implement the setter, or
                the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_num_max_queued_urbs'):
            return False
        try:
            return bool(self._driver.set_num_max_queued_urbs(value=value))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting NumMaxQueuedUrbs: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "NumMaxQueuedUrbs change failed",
                f"Could not set NumMaxQueuedUrbs to {value}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize (GigE-only Pylon node).

        Packet size in bytes. 1500 = standard Ethernet MTU; 9000 =
        typical jumbo-frame size. Larger packets reduce per-camera
        CPU + packet rate but require OS-level jumbo-frame config.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            size_bytes: Packet size in bytes (positive int).

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, size_bytes is non-positive, the driver
                doesn't implement the setter, or the node is not
                exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_gev_packet_size'):
            return False
        try:
            return bool(self._driver.set_gev_packet_size(size_bytes=size_bytes))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting GevSCPSPacketSize: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "GevSCPSPacketSize change failed",
                f"Could not set GevSCPSPacketSize to {size_bytes}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD (GigE inter-packet delay, in clock ticks).

        Inserts a wait between successive packets to throttle the
        camera. Used when multiple cameras share a single GigE link
        or when the host CPU can't keep up. 0 = no delay.

        USB3 cameras do not expose the node; returns False so the
        bench-probe sweep can call this method unconditionally per
        cell.

        Args:
            delay_ticks: Non-negative int; camera-specific tick rate.

        Returns:
            bool: True on success. False if the camera is absent /
                inactive, delay_ticks is negative, the driver doesn't
                implement the setter, or the node is not exposed.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return False
        if not hasattr(self._driver, 'set_gev_inter_packet_delay'):
            return False
        try:
            return bool(self._driver.set_gev_inter_packet_delay(
                delay_ticks=delay_ticks
            ))
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting GevSCPD: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "GevSCPD change failed",
                f"Could not set GevSCPD to {delay_ticks}: "
                f"{type(ex).__name__}: {ex}."
            )
            raise

    def set_gain_sync(self, gain, *, timeout=5) -> None:
        """Run ``set_gain`` through the camera_executor and block until done.

        Args:
            gain: Gain value in dB.
            timeout: Max seconds to wait for completion.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_gain_sync')
        task = IOTask(action=self.set_gain, args=(gain,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def set_exposure_sync(self, exposure, *, timeout=5) -> None:
        """Run ``set_exposure_time`` through the camera_executor and block.

        Args:
            exposure: Exposure time in milliseconds.
            timeout: Max seconds to wait for completion.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'set_exposure_sync')
        task = IOTask(action=self.set_exposure_time, args=(exposure,))
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def set_max_acquisition_frame_rate(
        self,
        enabled: bool,
        fps: float = 1.0,
    ) -> None:
        """Enable or disable the camera's acquisition frame-rate cap.

        Passthrough to the camera driver's ``set_max_acquisition_frame_rate``.
        When enabled, the camera will not produce frames faster than
        ``fps`` regardless of sensor-readout capability. Used by the
        manual-record path (#633 Stage 2C) to clamp video to the user's
        requested FPS, and by char-tool crash protection.

        Args:
            enabled: True to cap frame rate, False to remove the cap.
            fps: Target frame rate in fps when ``enabled=True``.
                Ignored when ``enabled=False``.

        Raises:
            HardwareError: Underlying SDK call failed in the driver.
        """
        if not self._driver or not self._driver.active:
            return
        if not hasattr(self._driver, 'set_max_acquisition_frame_rate'):
            logger.warning(
                f'[SCOPE API ] set_max_acquisition_frame_rate: '
                f'{type(self._driver).__name__} does not implement this method'
            )
            return
        try:
            self._driver.set_max_acquisition_frame_rate(enabled=enabled, fps=fps)
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] Error setting max_acquisition_frame_rate: {ex}"
            )
            from modules.notification_center import notifications
            notifications.error(
                "Camera",
                "Frame-rate cap change failed",
                f"Could not set frame-rate cap to enabled={enabled}, "
                f"fps={fps}: {type(ex).__name__}: {ex}. Camera may still be "
                f"at the previous setting."
            )
            raise

    # --- Getters ---
    def get_gain(self) -> float:
        """Get the current camera gain.

        Returns:
            float: Gain in dB, or -1 if camera inactive.
        """

        if not self._driver or not self._driver.active: return -1
        return self._driver.get_gain()

    def get_exposure_time(self) -> float:
        """Get the current camera exposure time.

        Returns:
            float: Exposure time in milliseconds, or 0 if camera inactive.
        """

        if not self._driver or not self._driver.active: return 0
        exposure = self._driver.get_exposure_t()
        return exposure

    def get_frame_size(self) -> dict | None:
        """Get the current camera frame size.

        Returns:
            dict | None: Contains 'width' and 'height' in pixels, or
                None if inactive.
        """

        if not self._driver or not self._driver.active: return
        return self._driver.get_frame_size()

    def get_pixel_format(self) -> str | None:
        """Get the current camera pixel format.

        Returns:
            str | None: Pixel format string (e.g. 'Mono8'), or None if inactive.
        """
        if not self._driver or not self._driver.active:
            return None
        return self._driver.get_pixel_format()

    def get_max_width(self) -> int:
        """Get the maximum pixel width of the camera sensor.

        Returns:
            int: Max width in pixels, or 0 if camera inactive.
        """
        if (not self._driver) or (not self._driver.active): return 0
        return self._driver.get_max_frame_size()['width']

    def get_max_height(self) -> int:
        """Get the maximum pixel height of the camera sensor.

        Returns:
            int: Max height in pixels, or 0 if camera inactive.
        """
        if (not self._driver) or (not self._driver.active): return 0
        return self._driver.get_max_frame_size()['height']

    def get_width(self) -> int:
        """Get the current frame width setting.

        Returns:
            int: Current width in pixels, or 0 if camera unavailable.
        """
        if not self._driver: return 0
        return self._driver.get_frame_size()['width']

    def get_height(self) -> int:
        """Get the current frame height setting.

        Returns:
            int: Current height in pixels, or 0 if camera unavailable.
        """
        if not self._driver: return 0
        return self._driver.get_frame_size()['height']

    def get_binning_size(self) -> int:
        """Get the current camera binning size.

        Returns:
            int: Current binning factor (1 if camera inactive).
        """
        if not self._driver or not self._driver.active:
            return 1

        return self._driver.get_binning_size()

    def get_supported_pixel_formats(self) -> tuple:
        """Get the list of supported camera pixel formats.

        Returns:
            tuple: Supported format strings, or empty tuple if inactive.
        """
        if not self._driver or not self._driver.active:
            return ()
        return self._driver.get_supported_pixel_formats()

    def get_available_binning_sizes(self) -> list:
        """Return list of binning sizes supported by connected camera.

        Returns:
            list: Supported binning factors (e.g. ``[1, 2, 4]``). Defaults
                to ``[1]`` if no camera is active.
        """
        if not self._driver or not self._driver.active:
            return [1]
        try:
            return self._driver.profile.binning_sizes
        except (AttributeError, TypeError):
            return [1]

    # --- Capture ---
    def capture(self, *args, **kwargs):
        return self._scope.capture(*args, **kwargs)

    def capture_complete(self, *args, **kwargs):
        return self._scope.capture_complete(*args, **kwargs)

    def capture_blocking(self) -> 'np.ndarray | bool | None':
        """Capture an image with illumination, blocking until the frame is ready. DEPRECATED.

        Deprecated: use ``capture_and_wait`` directly. Will be removed in
        a future release.

        Returns:
            numpy.ndarray | False | None: Captured image array, False on
                grab failure, or None if LED/camera are unavailable.
        """
        warnings.warn(
            "Lumascope.capture_blocking is deprecated. Use capture_and_wait() instead.",
            DeprecationWarning, stacklevel=2,
        )
        if not self._scope._led_driver: return
        if not self._driver or not self._driver.active: return

        return self.capture_and_wait()

    def capture_and_wait(self, *args, **kwargs):
        return self._scope.capture_and_wait(*args, **kwargs)

    def capture_and_wait_sync(self, *, timeout: float = 30, **kwargs) -> 'np.ndarray | bool | None':
        """Run ``capture_and_wait`` through the camera_executor and block.

        Args:
            timeout: Max seconds to wait for completion.
            **kwargs: Forwarded to ``capture_and_wait``.

        Returns:
            The captured image array, or None on failure.
        """
        ex = self._scope._require_executor(self._scope._camera_executor, 'capture_and_wait_sync')
        task = IOTask(action=self.capture_and_wait, kwargs=kwargs)
        fut = ex.put(task, return_future=True)
        if fut:
            return fut.result(timeout=timeout)
        return None

    def get_image(self, *args, **kwargs):
        return self._scope.get_image(*args, **kwargs)

    def get_image_with_chunks_from_buffer(self, *args, **kwargs):
        return self._scope.get_image_with_chunks_from_buffer(*args, **kwargs)

    def get_image_from_buffer(self, *args, **kwargs):
        return self._scope.get_image_from_buffer(*args, **kwargs)

    # --- State / lifecycle properties ---
    @property
    def camera_active(self) -> bool:
        return self._scope.camera_active

    def camera_is_connected(self) -> bool:
        """Check if the camera is active and connected.

        Returns:
            bool: True if camera is connected and active.
        """
        if not self._driver or not self._driver.active:
            return False

        return self._driver.is_connected()

    @property
    def camera_gain(self) -> float:
        return self._scope.camera_gain

    @property
    def camera_exposure_ms(self) -> float:
        return self._scope.camera_exposure_ms

    @property
    def camera_frame_size(self) -> dict:
        return self._scope.camera_frame_size

    @property
    def camera_max_frame_size(self) -> dict:
        return self._scope.camera_max_frame_size

    @property
    def camera_min_frame_size(self) -> dict:
        return self._scope.camera_min_frame_size

    @property
    def camera_max_exposure(self):
        return self._scope.camera_max_exposure

    @property
    def camera_max_gain(self):
        return self._scope.camera_max_gain

    @property
    def camera_pixel_format(self) -> str:
        return self._scope.camera_pixel_format

    # --- Save / restore ---
    def save_camera_state(self, tag: str) -> dict:
        """Snapshot the current camera gain and exposure for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_camera_state``.
        """
        gain = self.get_gain()
        exposure = self.get_exposure_time()
        snapshot = {'tag': tag, 'gain': gain, 'exposure': exposure}
        _api_log.info(f'save_camera_state tag={tag}: gain={gain} exp={exposure}')
        return snapshot

    def restore_camera_state(self, snapshot: dict) -> None:
        """Restore camera gain and exposure from a previously saved state.

        Args:
            snapshot: Return value from ``save_camera_state``.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        _api_log.info(f'restore_camera_state tag={tag}')
        gain = snapshot.get('gain', -1)
        exposure = snapshot.get('exposure', 0)
        if gain >= 0:
            self.set_gain(gain)
        if exposure > 0:
            self.set_exposure_time(exposure)

    # --- Camera config orchestration ---
    def apply_layer_camera_settings(self, gain: float, exposure_ms: float,
                                     auto_gain: bool = False,
                                     auto_gain_settings: dict | None = None) -> None:
        """Apply per-layer camera settings in a single batched call.

        Sets gain, exposure, and auto-gain state. Replaces 3 separate
        IOTask queues with a single call for atomicity.

        Args:
            gain: Camera gain in dB.
            exposure_ms: Exposure time in milliseconds.
            auto_gain: Whether auto-gain is enabled for this layer.
            auto_gain_settings: Dict with target_brightness, min_gain, max_gain
                               (required if auto_gain is True).
        """
        if not self._driver or not self._driver.active:
            return
        self.set_gain(gain)
        self.set_exposure_time(exposure_ms)
        if auto_gain_settings is not None:
            self.set_auto_gain(auto_gain, settings=auto_gain_settings)
        _api_log.info(f'apply_layer_camera_settings gain={gain}dB exp={exposure_ms}ms auto_gain={auto_gain}')

    def update_auto_gain_target_brightness(self, target_brightness: float) -> None:
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0 to 1.0).
        """
        if not self._driver or not self._driver.active:
            return
        self._driver.update_auto_gain_target_brightness(target_brightness)

    def auto_gain_once(self, state: bool, target_brightness: float,
                       min_gain: float, max_gain: float) -> None:
        """Run auto-gain for a single frame on the camera.

        Args:
            state: True to enable one-shot auto-gain.
            target_brightness: Target brightness (0.0 to 1.0).
            min_gain: Minimum gain in dB.
            max_gain: Maximum gain in dB.
        """
        if not self._driver or not self._driver.active:
            return
        self._driver.auto_gain_once(
            state=state,
            target_brightness=target_brightness,
            min_gain=min_gain,
            max_gain=max_gain,
        )

    def update_camera_config(self):
        """Context manager for batched camera config updates.

        Usage::

            with scope.update_camera_config():
                scope.set_gain(5.0)
                scope.set_exposure_time(100)

        Returns:
            A context manager. Falls back to ``contextlib.nullcontext()``
            when no camera is active.
        """
        if not self._driver or not self._driver.active:
            return contextlib.nullcontext()
        return self._driver.update_camera_config()

    def suppress_value_warnings(self, *args, **kwargs):
        return self._scope.suppress_value_warnings(*args, **kwargs)

    # --- Operation flags ---
    @property
    def is_capturing(self) -> bool:
        return self._scope.is_capturing

    @property
    def is_focusing(self) -> bool:
        return self._scope.is_focusing

    @property
    def capture_return(self):
        return self._scope.capture_return

    @property
    def autofocus_return(self):
        return self._scope.autofocus_return

    # --- Frame validity ---
    @property
    def frame_is_valid(self) -> bool:
        return self._scope.frame_is_valid

    def frames_until_valid(self, *args, **kwargs):
        return self._scope.frames_until_valid(*args, **kwargs)

    def count_frame(self, *args, **kwargs):
        return self._scope.count_frame(*args, **kwargs)

    # --- Scale bar ---
    @property
    def scale_bar_config(self) -> dict:
        return self._scope.scale_bar_config

    @property
    def scale_bar_enabled(self) -> bool:
        return self._scope.scale_bar_enabled

    def set_scale_bar(self, *args, **kwargs):
        return self._scope.set_scale_bar(*args, **kwargs)

    # --- Camera diagnostics (live, in-flight; cold probes live on DiagnosticsAPI) ---
    def get_camera_temps(self) -> dict:
        """Get camera temperature readings.

        Returns:
            dict: Mapping of sensor name to temperature in Celsius. Empty if inactive.
        """

        if not self._driver or not self._driver.active:
            return {}

        return self._driver.get_all_temperatures()

    def log_camera_temps(self) -> None:
        """Emit one INFO line per camera temperature sensor.

        No-op when no camera is connected. Called once on startup and
        periodically by ``start_camera_temp_logging``.
        """
        if not self.camera_is_connected():
            return
        for source, temp in self.get_camera_temps().items():
            logger.info(
                f'[CAM Class ] Camera {source} Temperature : {temp:.2f} degC')

    def start_camera_temp_logging(self, *args, **kwargs):
        return self._scope.start_camera_temp_logging(*args, **kwargs)

    def stop_camera_temp_logging(self, *args, **kwargs):
        return self._scope.stop_camera_temp_logging(*args, **kwargs)

    # --- Frame-flow listeners ---
    def add_camera_listener(self, *args, **kwargs):
        return self._scope.add_camera_listener(*args, **kwargs)

    def remove_camera_listener(self, *args, **kwargs):
        return self._scope.remove_camera_listener(*args, **kwargs)

    def register_frame_callback(self, cb) -> None:
        """Register a per-frame callback fired on every successful grab.

        Passthrough to the driver. Callback signature is
        ``cb(image, timestamp, chunks)``; runs on the SDK callback
        thread (Pylon ``PylonImageGrab`` / IDS grab loop / simulated
        pump). Callbacks MUST NOT block -- heavy work belongs on an
        executor. No-op when no camera is connected. Used by the
        manual-record path to drive saves on camera ticks instead of
        Kivy Clock.
        """
        if not self._driver or not self._driver.active:
            return
        try:
            self._driver.register_frame_callback(cb)
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] register_frame_callback failed: {ex}"
            )

    def unregister_frame_callback(self, cb) -> None:
        """Remove a callback registered via ``register_frame_callback``.

        Passthrough to the driver. No-op when no camera is connected
        or the callback was never registered.
        """
        if not self._driver:
            return
        try:
            self._driver.unregister_frame_callback(cb)
        except Exception as ex:
            logger.exception(
                f"[SCOPE API ] unregister_frame_callback failed: {ex}"
            )
