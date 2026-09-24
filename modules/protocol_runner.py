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
    pending = runner.run_single_scan(
        protocol,
        sequence_name="test_scan",
        image_capture_config=runner.build_image_capture_config(image_mode="8bit"),
    )
    result = pending.wait(timeout_s=300)     # or runner.wait_for_completion()
    print(result.status, result.reason, result.message)
"""

import pathlib
import typing

import modules.image_mode as image_mode_module
from modules.activity_claim import HeldClaim
from modules.exceptions import CaptureError, ConfigError
from modules.protocol import Protocol
from modules.run_outcome import PendingRunOutcome, RunOutcome
from modules.sequenced_capture_runner import (
    RunPlan,
    SequencedCaptureRunner,
    SequencedCaptureRunMode,
)

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
                'thread; build the session via ScopeSession.create, or '
                'inject protocol_thread at session construction.'
            )
        self.session = session
        self._protocol_thread = session.protocol_thread
        self._file_io_executor = session.file_io_executor
        self._executor = session.sequenced_capture_runner
        # The outcome of the last run THIS runner committed, and what
        # wait_for_completion answers from. None until a run commits, and
        # None again the moment a later call is refused: a refusal ran
        # nothing, so the previous run's result is not an answer about it.
        self._last_outcome: PendingRunOutcome | None = None

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
    ) -> PendingRunOutcome:
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

        Returns:
            The committed run's outcome. wait(timeout_s=...) on it for the
            status, reason, title and message the run ended with.

        Raises:
            ConfigError: image_capture_config was not provided -- there is
                no silent default image mode; the caller states the run's
                bit depth explicitly.
            ProtocolRunRefusedError: The run was refused before any state
                was committed; is_running() stays False and
                wait_for_completion() answers None.
        """
        return self._run(
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
    ) -> PendingRunOutcome:
        """Run a full protocol (multiple scans over time).

        Args:
            protocol: Protocol defining the steps, period, and duration
            sequence_name: Name for the output folder
            parent_dir: Parent directory for output
            image_capture_config: The run's capture/save intent; REQUIRED.
                Build one with build_image_capture_config(image_mode=...).
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions

        Returns:
            The committed run's outcome. wait(timeout_s=...) on it for the
            status, reason, title and message the run ended with.

        Raises:
            ConfigError: image_capture_config was not provided -- there is
                no silent default image mode; the caller states the run's
                bit depth explicitly.
            ProtocolRunRefusedError: The run was refused before any state
                was committed; is_running() stays False and
                wait_for_completion() answers None.
        """
        return self._run(
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

    def start_composite(
        self,
        sequence_name: str = 'composite',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        run_trigger_source: str = 'api_composite',
        engineering_mode: bool | None = None,
    ) -> PendingRunOutcome:
        """Assemble a composite run and launch it, returning once committed.

        Split out of run_composite so a caller that must not block -- a GUI
        click on the thread that draws the button -- gets the same assembly
        without the wait. The alternative was for such a caller to build the
        run config itself, which is the duplicate composite implementation
        this run kind exists to retire.

        Args:
            sequence_name: Name for the output folder.
            parent_dir: Parent directory for output. Defaults to
                'Manual/Composites' under the live folder, where the button
                already puts it, so a script's composite and a click's land
                in the same place.
            callbacks: Optional dict of callback functions.
            run_trigger_source: Provenance recorded on the run and named
                in refusals, so a GUI click records its own token rather
                than the API's.
            engineering_mode: Whether the run stamps the turret position
                into its filenames. A GUI caller passes its live flag, which
                a plugin may have flipped after the session was built; None
                reads the mode the session was built in.

        Returns:
            The run's outcome, to wait on or to ignore.

        Raises:
            ProtocolRunRefusedError: Fewer than two channels are set to
                capture an image, so nothing could be merged; or the
                runner refused the run itself (already running, files
                still writing, hardware not connected).
        """
        import modules.config_helpers as config_helpers

        settings = self.session.capture_settings_snapshot()
        input_config = config_helpers.get_composite_capture_config_from_settings(
            settings,
            self.session.objective_helper,
            position=self.session.get_current_plate_position(),
        )
        protocol = self.session.scope.protocols.create_protocol(input_config=input_config)

        if parent_dir is None:
            parent_dir = (
                pathlib.Path(settings.get('live_folder', '.')).resolve() / 'Manual' / 'Composites'
            )

        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_COMPOSITE,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=(
                config_helpers.get_composite_image_capture_config_from_settings(settings)
            ),
            enable_image_saving=True,
            callbacks=callbacks,
            # A composite is an interactive act on a scope the user is
            # standing at: it hands the illumination back the way it was
            # found, rather than forcing every channel dark the way an
            # unattended scan does.
            leds_state_at_end='return_to_original',
            composite_thresholds_percent=config_helpers.get_composite_blend_thresholds(settings),
            engineering_mode=engineering_mode,
        )

    def run_autofocus(
        self,
        layer: str,
        save_characterization_data: bool = False,
        sequence_name: str = 'autofocus',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        claim: HeldClaim | None = None,
    ) -> PendingRunOutcome:
        """Autofocus once on *layer*, at the current stage position.

        The headless twin of the standalone autofocus button: a
        one-position, one-layer run with autofocus on, saving no images and
        no run artifacts, that leaves the stage at the focus it found.

        The layer is named rather than discovered. A GUI reads it from
        whichever drawer is open, which is a fact about a running GUI and
        means nothing to a caller that has none, so this asks for it and
        has no default -- an autofocus on a layer nobody chose is not a
        useful answer.

        Everything else comes from the settings store, so this run and a
        click on the button resolve the same way.

        The stage is deliberately left where the focus was found, and there
        is no return-to-position input: ending at the focus is the point.
        A caller sweeping the same field repeatedly sets its own Z between
        runs, which it must do anyway for the measurements to be comparable.

        Args:
            layer: Which layer to focus on ('BF', 'Green', ...).
            save_characterization_data: Whether to write the per-position
                focus scores this sweep measured. Off by default: a caller
                that does not ask for data gets no folder. When on, the
                run's outcome reports whether the file was written and
                names it (af_data_saved / af_data_path), which is the only
                way a headless caller can tell a delivered file from a
                requested one.
            sequence_name: Name for the run.
            parent_dir: Where characterization data goes. Defaults to
                'Autofocus Characterization' under the live folder, where
                the button already puts it.
            callbacks: Optional dict of callback functions.
            claim: A claim the caller holds -- the one
                ``session.diagnostic_claim()`` yields -- to run under
                instead of taking the scope. The run acts inside the
                caller's activity and cannot release its claim; it is
                refused if that claim no longer holds. None takes the scope
                for this run alone.

        Returns:
            The run's outcome, to wait on or to ignore.

        Raises:
            ConfigError: *layer* is not a layer this release has.
            ProtocolRunRefusedError: The runner refused the request
                (already running, files still writing, hardware not
                connected); no state was committed.
        """
        import modules.config_helpers as config_helpers

        settings = self.session.capture_settings_snapshot()
        input_config = config_helpers.get_standalone_capture_config_from_settings(
            settings,
            self.session.objective_helper,
            self.session.wellplate_loader,
            layer=layer,
            position=self.session.get_current_plate_position(),
            position_name='Autofocus',
            autofocus=True,
            use_zstacking=False,
            # A standalone autofocus never pulses stimulation at the sample:
            # it is a measurement of focus, and firing the stim hardware
            # during one is something no caller has asked for.
            stim_config={},
        )
        protocol = self.session.scope.protocols.create_protocol(input_config=input_config)

        if parent_dir is None:
            parent_dir = (
                pathlib.Path(settings.get('live_folder', '.')).resolve()
                / 'Autofocus Characterization'
            )

        # Resolved here rather than left empty: the prepare boundary reads an
        # empty parent directory as "suppress artifacts", and the autofocus
        # engine raises outright when asked to save with nowhere to save to.
        # Suppressing artifacts and delivering data are not in conflict --
        # the run directory setup returns early on suppression while the
        # parent directory is still taken from the plan.
        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            run_trigger_source='api_autofocus',
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=config_helpers.get_image_capture_config_from_settings(settings),
            enable_image_saving=False,
            callbacks=callbacks,
            # A one-field operation at a scope someone is standing at, so it
            # hands the illumination back the way it was found rather than
            # forcing every channel dark the way a plate traverse does.
            leds_state_at_end='return_to_original',
            disable_saving_artifacts=True,
            save_autofocus_data=save_characterization_data,
            claim=claim,
        )

    def run_zstack(
        self,
        layer: str,
        sequence_name: str = 'zstack',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_start: bool = True,
        run_trigger_source: str = 'api_zstack',
        engineering_mode: bool | None = None,
        enable_image_saving: bool = True,
    ) -> PendingRunOutcome:
        """Capture a z-stack on *layer*, around the current stage position.

        The one implementation of a z-stack run, for a script and for the
        Acquire button on the z-stack panel alike: a one-position,
        one-layer run that expands into one step per slice. The slices are
        its product, so unlike an autofocus this one saves its images
        unless the caller says otherwise.

        The stack's range, step size and reference -- whether the current
        position is the top, centre or bottom of the sweep -- come from the
        settings store, as does everything else about the capture, so this
        run and a click on the button resolve the same way. The layer is named by the
        caller for the same reason run_autofocus asks for it: an open
        drawer is a fact about a running GUI.

        Autofocus is forced OFF for the step, matching the button. A
        z-stack with autofocus enabled refocuses at each slice and
        flattens the stack it was asked to capture.

        Stimulation configs are carried onto the step as stored, enabled or
        not. Stated rather than
        inherited silently: if the API should instead carry only the
        enabled ones, this is the line that changes.

        Args:
            layer: Which layer to capture ('BF', 'Green', ...).
            sequence_name: Name for the output folder.
            parent_dir: Parent directory for output. Defaults to
                'Manual/Z-Stacks' under the live folder, where the button
                already puts it.
            callbacks: Optional dict of callback functions.
            return_to_start: Whether to put the stage back where the stack
                was centred when the run ends. On by default, because a
                stack leaves Z at whichever end it finished on, which is
                not where the operator was looking. A bool rather than a
                position, so a caller cannot hand back coordinates in a
                frame this run never used.
            run_trigger_source: Provenance recorded on the run and named
                in refusals, so a GUI click records its own token rather
                than the API's.
            engineering_mode: Whether the run stamps the turret position
                into its filenames. A GUI caller passes its live flag, which
                a plugin may have flipped after the session was built; None
                reads the mode the session was built in.
            enable_image_saving: Whether the slices are written. On by
                default; the engineering panel's "disable image saving"
                switch is the one caller that turns it off, to exercise the
                stage without filling the disk.

        Returns:
            The run's outcome, to wait on or to ignore.

        Raises:
            ConfigError: *layer* is not a layer this release has.
            ObjectiveUnknownError: The objective in the light path is
                unknown, so no slice could say what it was taken with.
            ProtocolRunRefusedError: The runner refused the request
                (already running, files still writing, hardware not
                connected); no state was committed.
        """
        import modules.config_helpers as config_helpers

        settings = self.session.capture_settings_snapshot()
        position = self.session.get_current_plate_position()
        input_config = config_helpers.get_standalone_capture_config_from_settings(
            settings,
            self.session.objective_helper,
            self.session.wellplate_loader,
            layer=layer,
            position=position,
            position_name='ZStack',
            # Off, and not a caller's choice: an autofocus at every slice
            # re-centres the very range the stack is sweeping.
            autofocus=False,
            use_zstacking=True,
            # Unfiltered, matching the starter: every layer's stored config
            # rides along whether or not that layer is enabled.
            stim_config=config_helpers.get_stim_configs(settings),
        )
        protocol = self.session.scope.protocols.create_protocol(input_config=input_config)

        if parent_dir is None:
            parent_dir = (
                pathlib.Path(settings.get('live_folder', '.')).resolve() / 'Manual' / 'Z-Stacks'
            )

        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=config_helpers.get_image_capture_config_from_settings(settings),
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
            return_to_position=position if return_to_start else None,
            # A one-field operation at a scope someone is standing at, so it
            # hands the illumination back the way it was found.
            leds_state_at_end='return_to_original',
            engineering_mode=engineering_mode,
        )

    def run_composite(
        self,
        sequence_name: str = 'composite',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        merge_timeout_s: float = 900.0,
        engineering_mode: bool | None = None,
    ) -> str:
        """Capture one frame per acquiring channel and merge them.

        A composite is a single-position run through the same engine as
        every other run kind: one step per channel at the current stage
        position, each at its own stored focus, followed by the merge that
        combines them into one image.

        The channel set, the capture format and the depth all come from the
        settings snapshot rather than from any widget, so a headless caller
        gets the same run a GUI click does.

        The merged artifact is the run's real product, so this BLOCKS until
        the merge settles and returns where the artifact landed. A run that
        reported 'completed' while the merged file was missing would be
        indistinguishable from a successful one to every headless caller,
        which is the boundary this run kind exists to fix.

        Args:
            sequence_name: Name for the output folder.
            parent_dir: Parent directory for output. Defaults to
                'Manual/Composites' under the live folder, as start_composite.
            callbacks: Optional dict of callback functions.
            merge_timeout_s: Upper bound on the whole capture-and-merge
                wait. Covers the run itself, so it is longer than the
                merge's own internal drain bound.
            engineering_mode: Whether the run stamps the turret position
                into its filenames; None reads the mode the session was
                built in.

        Returns:
            The path of the merged composite.

        Raises:
            ProtocolRunRefusedError: Fewer than two channels are set to
                capture an image, so nothing could be merged; or the
                runner refused the run itself (already running, files
                still writing, hardware not connected).
            CaptureError: The run finished but produced no composite. The
                error names the machine-readable reason, so a caller can
                tell an aborted run from a failed merge from a timeout.
        """
        outcome = self.start_composite(
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            callbacks=callbacks,
            engineering_mode=engineering_mode,
        )
        settled = outcome.wait(timeout_s=merge_timeout_s)
        if settled is None:
            raise CaptureError(
                f'the composite did not report an outcome within '
                f'{merge_timeout_s:.0f}s; the run or its merge is wedged',
                'merge_timeout',
            )
        if not settled.merged:
            # A run that did not complete names its own ending; a completed
            # run with no artifact names what the merge did. One field is
            # empty in each case, so the caller never has to guess which
            # vocabulary it is reading.
            code = settled.merge_reason or settled.reason
            raise CaptureError(f'no composite was produced ({code})', code)
        return settled.artifact_path

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
        leds_state_at_end: str = 'off',
        composite_thresholds_percent: dict | None = None,
        engineering_mode: bool | None = None,
        disable_saving_artifacts: bool = False,
        save_autofocus_data: bool = False,
        claim: HeldClaim | None = None,
    ) -> PendingRunOutcome:
        """Internal: configure and launch the sequenced capture executor.

        Returns:
            The committed run's outcome.

        Raises:
            ConfigError: image_capture_config was not provided; raised
                before any executor starts or hardware moves.
            ProtocolRunRefusedError: The runner refused the request (already
                running, files still writing, empty/invalid protocol,
                hardware not connected); no state was committed and the
                user was already notified once.
        """
        # Cleared before the gate, stored only once a run has committed:
        # whatever this call does, wait_for_completion must not go on
        # answering with the previous run's result.
        self._last_outcome = None

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

        # The session's as-built mode is the default; only a caller holding
        # a LIVE flag (the GUI, whose plugin flips it after the session
        # exists) has a reason to say otherwise.
        if engineering_mode is None:
            engineering_mode = self.session.engineering_mode

        # Copied so the engine cannot mutate the caller's dict.
        run_callbacks = dict(callbacks or {})

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
            callbacks=run_callbacks,
            return_to_position=return_to_position,
            leds_state_at_end=leds_state_at_end,
            composite_thresholds_percent=composite_thresholds_percent,
            engineering_mode=engineering_mode,
            # Forwarded with the boundary's own names and its own defaults,
            # so this helper and the prepare it wraps stay one-to-one. No
            # existing caller passes either; both were reachable only from
            # inside the engine until a run kind needed to ask for them.
            disable_saving_artifacts=disable_saving_artifacts,
            save_autofocus_data=save_autofocus_data,
            borrowed_claim=claim.lend() if claim is not None else None,
            autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                self.session.settings, self.session.settings_lock
            ),
            **config_helpers.get_sequenced_run_settings(self.session.settings, run_mode=run_mode),
        )

        # Run-state truth is the session claim, committed inside
        # start()'s gate-and-commit -- a refusal means no state changed,
        # and leaves _last_outcome None so a caller that waits is told
        # "nothing ran" rather than blocked on a run that never started.
        # The local is what this call returns: reading the attribute back
        # is a race, because a run that fails at start releases the claim
        # synchronously and a rival can commit in between.
        outcome = self._executor.start(plan)
        self._last_outcome = outcome
        return outcome

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._executor.run_in_progress()

    def run_dir(self) -> pathlib.Path | None:
        return self._executor.run_dir()

    def run_trigger_source(self) -> 'str | None':
        """The trigger kind of the run holding the scope; None when no
        run holds it."""
        return self._executor.run_trigger_source()

    def run_outcome(self) -> PendingRunOutcome | None:
        """The live or last run's handle -- what abort() names to stop it."""
        return self._executor.run_outcome()

    def is_live_run(self, run: PendingRunOutcome | None) -> bool:
        """Whether *run*, a handle a run call returned, is the live run."""
        return self._executor.is_live_run(run)

    def remaining_scans(self) -> int:
        return self._executor.remaining_scans()

    def protocol_interval(self):
        """The loaded protocol's scan period; None before the first run."""
        return self._executor.protocol_interval()

    def current_step_color(self) -> 'str | None':
        return self._executor.current_step_color()

    @property
    def video_pending_writes(self) -> int:
        """Frames across the run's video steps not yet on disk; read, not called."""
        return self._executor.video_pending_writes

    def discard_video_pending(self) -> None:
        self._executor.discard_video_pending()

    def prepare(self, **kwargs):
        """Forward to the engine's prepare(); returns the RunPlan.

        For callers that need the two-phase prepare/start seam directly
        (run_single_scan / run_protocol wrap it with config assembly).
        """
        return self._executor.prepare(**kwargs)

    def start(self, plan: RunPlan) -> PendingRunOutcome:
        """Forward to the engine's start() -- the commitment point.

        Records the committed run as this runner's last, so a caller that
        drove prepare/start directly still has wait_for_completion.
        """
        self._last_outcome = None
        outcome = self._executor.start(plan)
        self._last_outcome = outcome
        return outcome

    def wait_for_run_idle(self, timeout_s: float) -> bool:
        """Block until the engine's cleanup fully lands (claim released).

        Distinct from wait_for_completion, which answers with the run's
        outcome: this one answers only whether the runner is idle, for a
        caller about to start something else."""
        return self._executor.wait_for_run_idle(timeout_s)

    def abort(self, run: PendingRunOutcome | None) -> None:
        """Abort *run*, the handle a run call returned.

        Anyone may stop the live run. A handle naming a run that has ended
        raises out of reset() below -- RunAlreadyEndedError when nothing is
        live, the 'run_not_live' refusal when another run is -- ahead of
        every side effect here, because a
        refused abort must leave the protocol thread running and its
        waiters waiting. Ordering is the guard: the thread signal is
        unconditional once reset() has returned.

        Waiters are NOT released here. The run's outcome settles inside
        cleanup's finally, so a caller that wakes from
        wait_for_completion knows the teardown happened rather than only
        that someone asked for it.
        """
        self._executor.reset(run)
        self._protocol_thread.abort()

    def wait_for_completion(self, timeout: float | None = None) -> RunOutcome | None:
        """How did the last run this runner committed end?

        Blocks until that run settles, then returns its outcome: the
        status and reason it ended with, the title and message a user
        would read, and what the merge produced.

        None when the bound expires, and None AT ONCE when the last call
        was refused or no run has ever been committed -- a refused start
        ran nothing, and answering with an earlier run's 'completed'
        would be a stale answer to a question about this one.
        """
        outcome = self._last_outcome
        if outcome is None:
            return None
        return outcome.wait(timeout_s=timeout)
