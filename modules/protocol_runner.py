# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
GUI-independent protocol runner.

Provides a clean API for running protocols (scans, full protocols, autofocus)
without any Kivy/GUI dependencies. Used by the REST API and standalone scripts.
The LumaViewPro GUI continues to use ProtocolSettings for UI orchestration,
but both ultimately delegate to SequencedCaptureExecutor.

Usage
-----
    from modules.scope_session import ScopeSession
    from modules.protocol_runner import ProtocolRunner

    session = ScopeSession.create(settings=settings)
    runner = ProtocolRunner(session)

    protocol = Protocol.from_file("my_protocol.csv")
    runner.run_single_scan(protocol, sequence_name="test_scan")
    runner.wait_for_completion()
"""

import datetime
import pathlib
import threading
import typing

from lvp_logger import logger
import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.sequenced_capture_executor import SequencedCaptureExecutor, SequencedCaptureRunMode
from modules.sequential_io_executor import SequentialIOExecutor
from modules.autofocus_executor import AutofocusExecutor


class ProtocolRunner:
    """GUI-independent protocol runner wrapping SequencedCaptureExecutor."""

    def __init__(
        self,
        session,
        protocol_executor: SequentialIOExecutor | None = None,
        file_io_executor: SequentialIOExecutor | None = None,
        autofocus_io_executor: SequentialIOExecutor | None = None,
    ):
        """
        Args:
            session: ScopeSession instance providing scope, settings, executors
            protocol_executor: Executor for protocol sequencing (created if None)
            file_io_executor: Executor for file I/O (created if None)
            autofocus_io_executor: Executor for autofocus (created if None)
        """
        self.session = session

        self._protocol_executor = protocol_executor or SequentialIOExecutor(name="PROTOCOL")
        self._file_io_executor = file_io_executor or SequentialIOExecutor(name="FILE")
        self._autofocus_io_executor = autofocus_io_executor or SequentialIOExecutor(name="AUTOFOCUS")

        self._completion_event = threading.Event()

        self._executor = SequencedCaptureExecutor(
            scope=session.scope,
            stage_offset=session.settings.get('stage_offset', {}),
            io_executor=session.io_executor,
            protocol_executor=self._protocol_executor,
            file_io_executor=self._file_io_executor,
            camera_executor=session.camera_executor,
            autofocus_io_executor=self._autofocus_io_executor,
        )

        self._owned_executors_started = False

    @property
    def sequenced_capture_executor(self) -> SequencedCaptureExecutor:
        return self._executor

    # ------------------------------------------------------------------
    # Config helpers (pure — no GUI reads)
    # ------------------------------------------------------------------

    def build_image_capture_config(
        self,
        live_format: str = "TIFF",
        sequenced_format: str = "TIFF",
        use_full_pixel_depth: bool = True,
    ) -> dict:
        """Build an image capture config dict without reading from GUI."""
        return {
            'output_format': {
                'live': live_format,
                'sequenced': sequenced_format,
            },
            'use_full_pixel_depth': use_full_pixel_depth,
        }

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_single_scan(
        self,
        protocol: Protocol,
        sequence_name: str = "scan",
        parent_dir: pathlib.Path | str | None = None,
        image_capture_config: dict | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
    ):
        """Run a single scan through the protocol steps.

        Args:
            protocol: Protocol defining the steps to execute
            sequence_name: Name for the output folder
            parent_dir: Parent directory for output (defaults to settings['live_folder']/ProtocolData)
            image_capture_config: Image format config (defaults to TIFF)
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions
            return_to_position: Optional position to return to after scan
        """
        self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            run_trigger_source='api_scan',
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
            return_to_position=return_to_position,
        )

    def run_protocol(
        self,
        protocol: Protocol,
        sequence_name: str = "protocol",
        parent_dir: pathlib.Path | str | None = None,
        image_capture_config: dict | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
    ):
        """Run a full protocol (multiple scans over time).

        Args:
            protocol: Protocol defining the steps, period, and duration
            sequence_name: Name for the output folder
            parent_dir: Parent directory for output
            image_capture_config: Image format config (defaults to TIFF)
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions
        """
        self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            run_trigger_source='api_protocol',
            max_scans=None,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
        )

    def _run(
        self,
        protocol: Protocol,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        sequence_name: str,
        parent_dir: pathlib.Path | str | None = None,
        image_capture_config: dict | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
    ):
        """Internal: configure and launch the sequenced capture executor."""
        self._ensure_executors_started()
        self._completion_event.clear()

        if parent_dir is None:
            parent_dir = pathlib.Path(
                self.session.settings.get('live_folder', '.')
            ).resolve() / "ProtocolData"
        else:
            parent_dir = pathlib.Path(parent_dir)

        if image_capture_config is None:
            image_capture_config = self.build_image_capture_config()

        import modules.config_helpers as config_helpers
        autogain_settings = config_helpers.get_auto_gain_settings(self.session.settings)

        merged_callbacks = dict(callbacks or {})
        # Wire up a completion callback
        user_complete = merged_callbacks.get('run_complete')

        def _on_complete(**kwargs):
            if user_complete:
                user_complete(**kwargs)
            self._completion_event.set()
            self.session.protocol_running.clear()

        merged_callbacks['run_complete'] = _on_complete

        # Restore autofocus via settings dict (safe: called on protocol thread completion)
        settings = self.session.settings
        merged_callbacks.setdefault(
            'restore_autofocus_state',
            lambda layer, value: settings[layer].__setitem__('autofocus', value),
        )

        # Snapshot autofocus states before handing off to the protocol thread
        initial_autofocus_states = {
            layer: settings[layer]['autofocus']
            for layer in common_utils.get_layers()
            if layer in settings
        }

        self.session.protocol_running.set()

        self._executor.run(
            protocol=protocol,
            run_mode=run_mode,
            run_trigger_source=run_trigger_source,
            max_scans=max_scans,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=enable_image_saving,
            autogain_settings=autogain_settings,
            callbacks=merged_callbacks,
            return_to_position=return_to_position,
            leds_state_at_end="off",
            video_as_frames=self.session.settings.get('video_as_frames', False),
            initial_autofocus_states=initial_autofocus_states,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._executor.run_in_progress()

    def run_dir(self) -> pathlib.Path | None:
        return self._executor.run_dir()

    def abort(self):
        """Abort the current run."""
        self._executor.reset()
        self._completion_event.set()
        self.session.protocol_running.clear()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Block until the run completes. Returns True if completed, False on timeout."""
        return self._completion_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_executors_started(self):
        """Start any executors we created if not already started."""
        if not self._owned_executors_started:
            self._protocol_executor.start()
            self._file_io_executor.start()
            self._autofocus_io_executor.start()
            self._owned_executors_started = True

    def shutdown(self):
        """Shut down executors that we created."""
        if self._owned_executors_started:
            self._protocol_executor.shutdown()
            self._file_io_executor.shutdown()
            self._autofocus_io_executor.shutdown()
            self._owned_executors_started = False
