# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
GUI-independent protocol runner.

Provides a clean API for running protocols (scans, full protocols, autofocus)
without any Kivy/GUI dependencies. Used by the REST API and standalone scripts.
The LumaViewPro GUI continues to use ProtocolSettings for UI orchestration,
but both ultimately delegate to SequencedCaptureRunner.

Usage
-----
    from modules.scope_session import ScopeSession
    from modules.protocol_runner import ProtocolRunner

    session = ScopeSession.create(settings=settings)
    runner = ProtocolRunner(session)

    protocol = Protocol.from_file("my_protocol.csv")
    runner.run_single_scan(
        protocol,
        sequence_name="test_scan",
        image_capture_config=runner.build_image_capture_config(image_mode="8bit"),
    )
    runner.wait_for_completion()
"""

import pathlib
import threading
import typing

import modules.common_utils as common_utils
import modules.image_mode as image_mode_module
from modules.exceptions import ConfigError, ProtocolRunRefusedError
from modules.protocol import Protocol
from modules.sequenced_capture_runner import SequencedCaptureRunner, SequencedCaptureRunMode

from lvp_logger import logger


class ProtocolRunner:
    """GUI-independent protocol runner wrapping the session's
    SequencedCaptureRunner.

    One engine per session: the runner wraps the session-composed
    instance rather than constructing a second one, so the GUI, L2, and
    REST all drive (and observe) the same run state behind the same
    claim.
    """

    def __init__(self, session):
        """
        Args:
            session: ScopeSession providing the composed engine, scope,
                settings, and executors. Must carry a protocol thread
                (the factories and the GUI host both compose one); a
                bare session cannot drive a scan loop.
        """
        if session.protocol_thread is None:
            raise RuntimeError(
                'ProtocolRunner requires a session composed with a protocol '
                'thread; build the session via ScopeSession.create / '
                'create_headless, or inject protocol_thread at session '
                'construction.'
            )
        self.session = session
        self._protocol_thread = session.protocol_thread
        self._file_io_executor = session.file_io_executor
        self._completion_event = threading.Event()
        self._executor = session.sequenced_capture_runner

    @property
    def sequenced_capture_runner(self) -> SequencedCaptureRunner:
        return self._executor

    # ------------------------------------------------------------------
    # Config helpers (pure -- no GUI reads)
    # ------------------------------------------------------------------

    def build_image_capture_config(
        self,
        *,
        image_mode: str,
        live_format: str = 'TIFF',
        sequenced_format: str = 'TIFF',
        jpg_quality: int = 90,
    ) -> image_mode_module.ImageCaptureConfig:
        """Build an image capture config without reading from GUI.

        image_mode is required: a headless run is a deliberate act by a
        script author, and an unstated mode silently decided the science
        data's bit depth (a script that captured full depth on older
        releases would quietly produce 8-bit files). capture_depth and
        save_encoding are derived together from the one image_mode value
        rather than carried independently, so the config that drives capture
        also drives the save: a 12-bit-scaled capture cannot be paired with
        an 8-bit save that stores it right-aligned (dark). This is the
        GUI-less mirror of get_image_capture_config_from_ui; both route
        through the same one constructor so the two paths cannot drift.
        """
        return image_mode_module.ImageCaptureConfig.from_image_mode(
            image_mode,
            output_format_live=live_format,
            output_format_sequenced=sequenced_format,
            jpg_quality=jpg_quality,
        )

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_single_scan(
        self,
        protocol: Protocol,
        sequence_name: str = 'scan',
        parent_dir: pathlib.Path | str | None = None,
        image_capture_config: image_mode_module.ImageCaptureConfig | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
    ):
        """Run a single scan through the protocol steps.

        Args:
            protocol: Protocol defining the steps to execute
            sequence_name: Name for the output folder
            parent_dir: Parent directory for output (defaults to settings['live_folder']/ProtocolData)
            image_capture_config: The run's capture/save intent; REQUIRED.
                Build one with build_image_capture_config(image_mode=...).
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions
            return_to_position: Optional position to return to after scan

        Raises:
            ConfigError: image_capture_config was not provided -- there is
                no silent default image mode; the caller states the run's
                bit depth explicitly.
            ProtocolRunRefusedError: The run was refused before any state
                was committed; is_running() stays False and
                wait_for_completion() is not armed.
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
        sequence_name: str = 'protocol',
        parent_dir: pathlib.Path | str | None = None,
        image_capture_config: image_mode_module.ImageCaptureConfig | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
    ):
        """Run a full protocol (multiple scans over time).

        Args:
            protocol: Protocol defining the steps, period, and duration
            sequence_name: Name for the output folder
            parent_dir: Parent directory for output
            image_capture_config: The run's capture/save intent; REQUIRED.
                Build one with build_image_capture_config(image_mode=...).
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions

        Raises:
            ConfigError: image_capture_config was not provided -- there is
                no silent default image mode; the caller states the run's
                bit depth explicitly.
            ProtocolRunRefusedError: The run was refused before any state
                was committed; is_running() stays False and
                wait_for_completion() is not armed.
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
        image_capture_config: image_mode_module.ImageCaptureConfig | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
    ):
        """Internal: configure and launch the sequenced capture executor.

        Raises:
            ConfigError: image_capture_config was not provided; raised
                before any executor starts or hardware moves.
            ProtocolRunRefusedError: The runner refused the request (already
                running, files still writing, empty/invalid protocol,
                hardware not connected); no state was committed and the
                user was already notified once.
        """
        # No silent default: an unstated image mode silently decided the
        # data's bit depth (an older-release script that captured full depth
        # would quietly produce 8-bit files). The caller states intent once;
        # this raises before any executor starts or hardware moves.
        if image_capture_config is None:
            raise ConfigError(
                'image_capture_config is required for a headless run: pass '
                'image_capture_config=runner.build_image_capture_config('
                "image_mode='8bit') (or one of the 12-bit modes) so the "
                "run's capture depth and save encoding are explicit."
            )

        if parent_dir is None:
            parent_dir = (
                pathlib.Path(self.session.settings.get('live_folder', '.')).resolve()
                / 'ProtocolData'
            )
        else:
            parent_dir = pathlib.Path(parent_dir)

        # One self-describing record per scan: the per-frame save path runs
        # thousands of times per session and cannot log its depth at info
        # level, so a scan's capture depth / on-disk encoding is otherwise
        # recoverable only by inspecting the output file tags afterward. This
        # line lets a support bundle state the mode the scan ran in.
        logger.info(
            f'[Protocol] scan "{sequence_name}" '
            f'image_mode={image_capture_config.image_mode} '
            f'capture_depth={image_capture_config.capture_depth} '
            f'save_encoding={image_capture_config.save_encoding}'
        )

        import modules.config_helpers as config_helpers

        autogain_settings = config_helpers.get_auto_gain_settings(self.session.settings)

        merged_callbacks = dict(callbacks or {})
        # Wire up a completion callback
        user_complete = merged_callbacks.get('run_complete')

        def _on_complete(**kwargs):
            if user_complete:
                user_complete(**kwargs)
            self._completion_event.set()

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

        plan = self._executor.prepare(
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
            leds_state_at_end='off',
            initial_autofocus_states=initial_autofocus_states,
            **config_helpers.get_sequenced_run_settings(self.session.settings),
        )

        # Run-state truth is the session claim, committed inside
        # start()'s gate-and-commit -- a refusal means no state changed.
        # The completion event is caller convenience, re-armed here and
        # restored on a start()-stage refusal: for the already-running
        # race the prior state was cleared (a live rival run resolves it
        # when its run_complete fires -- every _run wires the same
        # shared event), and for a claim refusal (e.g. a recording
        # holds the scope) the prior state was set, so restoring it lets
        # wait_for_completion return immediately instead of hanging on a
        # run that never started.
        completion_was_set = self._completion_event.is_set()
        self._completion_event.clear()
        try:
            self._executor.start(plan)
        except ProtocolRunRefusedError:
            if completion_was_set:
                self._completion_event.set()
            raise

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._executor.run_in_progress()

    def run_dir(self) -> pathlib.Path | None:
        return self._executor.run_dir()

    def run_trigger_source(self) -> 'str | None':
        """The current (or, between runs, most recent) run's trigger kind."""
        return self._executor.run_trigger_source()

    def remaining_scans(self) -> int:
        return self._executor.remaining_scans()

    def protocol_interval(self):
        """The loaded protocol's scan period; None before the first run."""
        return self._executor.protocol_interval()

    def current_step_color(self) -> 'str | None':
        return self._executor.current_step_color()

    def video_drain_busy(self) -> bool:
        return self._executor.video_drain_busy()

    def video_pending_writes(self) -> int:
        return self._executor.video_pending_writes()

    def discard_video_pending(self) -> None:
        self._executor.discard_video_pending()

    def prepare(self, **kwargs):
        """Forward to the engine's prepare(); returns the RunPlan.

        For callers that need the two-phase prepare/start seam directly
        (run_single_scan / run_protocol wrap it with config assembly).
        """
        return self._executor.prepare(**kwargs)

    def start(self, plan) -> None:
        """Forward to the engine's start() -- the commitment point."""
        self._executor.start(plan)

    def reset(self) -> None:
        """Unwind the current run without tearing the runner down.

        Distinct from abort(): reset() leaves the completion event and
        protocol thread alone (abort-and-continue); abort() also aborts
        the scan loop and resolves waiters (abort-and-teardown for this
        run's callers).
        """
        self._executor.reset()

    def wait_for_run_idle(self, timeout_s: float) -> bool:
        """Block until the engine's cleanup fully lands (claim released),
        not merely until run_complete fires -- the completion-event wait
        (wait_for_completion) resolves at the run-complete callback,
        moments before cleanup's end."""
        return self._executor.wait_for_run_idle(timeout_s)

    def set_scope(self, scope) -> None:
        """Rewire onto a new scope via the session (whole-cluster rewire)."""
        self.session.set_scope(scope)

    def abort(self):
        """Abort the current run."""
        self._protocol_thread.abort()
        self._executor.reset()
        self._completion_event.set()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Block until the run completes. Returns True if completed, False on timeout."""
        return self._completion_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Retained for callers that paired shutdown() with
        create_protocol_runner(); the engine, threads, and executors are
        session composition now, torn down by session.shutdown()."""
