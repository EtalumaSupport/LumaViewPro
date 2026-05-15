# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ImagingAPI -- sub-API for camera capture / image acquisition.

Phase 1 of Wave 7 decomposition. Thin delegating facade over the
Lumascope composition root. Bodies still live on Lumascope; later
phases relocate them and migrate callers.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.3 for the canonical
method list. Frame-listener registry and live_processing plugin
infrastructure ship in Wave 7 Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.camera import Camera


class ImagingAPI:
    """Imaging sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope', driver: 'Camera | None') -> None:
        self._scope = scope
        self._driver = driver

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

    def set_acquisition_stop_mode(self, *args, **kwargs):
        return self._scope.set_acquisition_stop_mode(*args, **kwargs)

    def set_bandwidth_reserve_mode(self, *args, **kwargs):
        return self._scope.set_bandwidth_reserve_mode(*args, **kwargs)

    def set_device_link_throughput_limit(self, *args, **kwargs):
        return self._scope.set_device_link_throughput_limit(*args, **kwargs)

    def set_max_transfer_size(self, *args, **kwargs):
        return self._scope.set_max_transfer_size(*args, **kwargs)

    def set_num_max_queued_urbs(self, *args, **kwargs):
        return self._scope.set_num_max_queued_urbs(*args, **kwargs)

    def set_gev_packet_size(self, *args, **kwargs):
        return self._scope.set_gev_packet_size(*args, **kwargs)

    def set_gev_inter_packet_delay(self, *args, **kwargs):
        return self._scope.set_gev_inter_packet_delay(*args, **kwargs)

    def set_gain_sync(self, *args, **kwargs):
        return self._scope.set_gain_sync(*args, **kwargs)

    def set_exposure_sync(self, *args, **kwargs):
        return self._scope.set_exposure_sync(*args, **kwargs)

    def set_max_acquisition_frame_rate(self, *args, **kwargs):
        return self._scope.set_max_acquisition_frame_rate(*args, **kwargs)

    # --- Getters ---
    def get_gain(self, *args, **kwargs):
        return self._scope.get_gain(*args, **kwargs)

    def get_exposure_time(self, *args, **kwargs):
        return self._scope.get_exposure_time(*args, **kwargs)

    def get_frame_size(self, *args, **kwargs):
        return self._scope.get_frame_size(*args, **kwargs)

    def get_pixel_format(self, *args, **kwargs):
        return self._scope.get_pixel_format(*args, **kwargs)

    def get_max_width(self, *args, **kwargs):
        return self._scope.get_max_width(*args, **kwargs)

    def get_max_height(self, *args, **kwargs):
        return self._scope.get_max_height(*args, **kwargs)

    def get_width(self, *args, **kwargs):
        return self._scope.get_width(*args, **kwargs)

    def get_height(self, *args, **kwargs):
        return self._scope.get_height(*args, **kwargs)

    def get_binning_size(self, *args, **kwargs):
        return self._scope.get_binning_size(*args, **kwargs)

    def get_supported_pixel_formats(self, *args, **kwargs):
        return self._scope.get_supported_pixel_formats(*args, **kwargs)

    def get_available_binning_sizes(self, *args, **kwargs):
        return self._scope.get_available_binning_sizes(*args, **kwargs)

    # --- Capture ---
    def capture(self, *args, **kwargs):
        return self._scope.capture(*args, **kwargs)

    def capture_complete(self, *args, **kwargs):
        return self._scope.capture_complete(*args, **kwargs)

    def capture_blocking(self, *args, **kwargs):
        return self._scope.capture_blocking(*args, **kwargs)

    def capture_and_wait(self, *args, **kwargs):
        return self._scope.capture_and_wait(*args, **kwargs)

    def capture_and_wait_sync(self, *args, **kwargs):
        return self._scope.capture_and_wait_sync(*args, **kwargs)

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
        return self._scope.camera_is_connected()

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
    def save_camera_state(self, *args, **kwargs):
        return self._scope.save_camera_state(*args, **kwargs)

    def restore_camera_state(self, *args, **kwargs):
        return self._scope.restore_camera_state(*args, **kwargs)

    # --- Camera config orchestration ---
    def apply_layer_camera_settings(self, *args, **kwargs):
        return self._scope.apply_layer_camera_settings(*args, **kwargs)

    def update_auto_gain_target_brightness(self, *args, **kwargs):
        return self._scope.update_auto_gain_target_brightness(*args, **kwargs)

    def auto_gain_once(self, *args, **kwargs):
        return self._scope.auto_gain_once(*args, **kwargs)

    def update_camera_config(self, *args, **kwargs):
        return self._scope.update_camera_config(*args, **kwargs)

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
    def get_camera_temps(self, *args, **kwargs):
        return self._scope.get_camera_temps(*args, **kwargs)

    def log_camera_temps(self, *args, **kwargs):
        return self._scope.log_camera_temps(*args, **kwargs)

    def start_camera_temp_logging(self, *args, **kwargs):
        return self._scope.start_camera_temp_logging(*args, **kwargs)

    def stop_camera_temp_logging(self, *args, **kwargs):
        return self._scope.stop_camera_temp_logging(*args, **kwargs)

    # --- Frame-flow listeners ---
    def add_camera_listener(self, *args, **kwargs):
        return self._scope.add_camera_listener(*args, **kwargs)

    def remove_camera_listener(self, *args, **kwargs):
        return self._scope.remove_camera_listener(*args, **kwargs)

    def register_frame_callback(self, *args, **kwargs):
        return self._scope.register_frame_callback(*args, **kwargs)

    def unregister_frame_callback(self, *args, **kwargs):
        return self._scope.unregister_frame_callback(*args, **kwargs)
