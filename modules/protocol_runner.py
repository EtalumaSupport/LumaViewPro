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
    pending = runner.run_single_scan(protocol, sequence_name="test_scan")
    result = pending.wait(timeout_s=300)     # pending.stop() stops it
    print(result.status, result.reason, result.message)
"""

import pathlib
import typing

from modules.activity_claim import HeldClaim
from modules.exceptions import CaptureError
from modules.protocol import Protocol
from modules.run_outcome import RunOutcome
from modules.sequenced_capture_runner import (
    RunHandle,
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
        self._executor = session.sequenced_capture_runner

    @property
    def sequenced_capture_runner(self) -> SequencedCaptureRunner:
        return self._executor

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_single_scan(
        self,
        protocol: Protocol,
        sequence_name: str = 'scan',
        parent_dir: pathlib.Path | str | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
        run_trigger_source: str = 'api_scan',
        engineering_mode: bool | None = None,
    ) -> RunHandle:
        """Run a single scan through the protocol steps.

        Args:
            protocol: Protocol defining the steps to execute
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path
            parent_dir: Parent directory for output (defaults to settings['live_folder']/ProtocolData)
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions
            return_to_position: Optional position to return to after scan
            run_trigger_source: Provenance recorded on the run and named
                in refusals, so the protocol panel's button records its own
                token rather than the API's.
            engineering_mode: Whether the run stamps the turret position
                into its filenames. A GUI caller passes its live flag, which
                a plugin may have flipped after the session was built; None
                reads the mode the session was built in.

        Returns:
            The committed run's handle. wait(timeout_s=...) on it for the
            outcome: the status, reason, title and message the run ended with.

        Raises:
            ProtocolRunRefusedError: The run was refused before any state
                was committed; no handle is returned and
                session.is_protocol_running stays False.
        """
        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
            return_to_position=return_to_position,
            engineering_mode=engineering_mode,
        )

    def run_protocol(
        self,
        protocol: Protocol,
        sequence_name: str = 'protocol',
        parent_dir: pathlib.Path | str | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        run_trigger_source: str = 'api_protocol',
        engineering_mode: bool | None = None,
    ) -> RunHandle:
        """Run a full protocol (multiple scans over time).

        Args:
            protocol: Protocol defining the steps, period, and duration
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path
            parent_dir: Parent directory for output
            enable_image_saving: Whether to save captured images
            callbacks: Optional dict of callback functions
            run_trigger_source: Provenance recorded on the run and named
                in refusals, so the protocol panel's button records its own
                token rather than the API's.
            engineering_mode: Whether the run stamps the turret position
                into its filenames. A GUI caller passes its live flag, which
                a plugin may have flipped after the session was built; None
                reads the mode the session was built in.

        Returns:
            The committed run's handle. wait(timeout_s=...) on it for the
            outcome: the status, reason, title and message the run ended with.

        Raises:
            ProtocolRunRefusedError: The run was refused before any state
                was committed; no handle is returned and
                session.is_protocol_running stays False.
        """
        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            run_trigger_source=run_trigger_source,
            max_scans=None,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
            engineering_mode=engineering_mode,
        )

    def start_composite(
        self,
        sequence_name: str = 'composite',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        run_trigger_source: str = 'api_composite',
        engineering_mode: bool | None = None,
    ) -> RunHandle:
        """Assemble a composite run and launch it, returning once committed.

        Split out of run_composite so a caller that must not block -- a GUI
        click on the thread that draws the button -- gets the same assembly
        without the wait. The alternative was for such a caller to build the
        run config itself, which is the duplicate composite implementation
        this run kind exists to retire.

        Args:
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path.
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
            The committed run's handle, to wait on or to ignore;
            wait(timeout_s=...) on it gives the run's outcome.

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
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'Manual' / 'Composites'

        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_COMPOSITE,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            enable_image_saving=True,
            callbacks=callbacks,
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
        run_trigger_source: str = 'api_autofocus',
        engineering_mode: bool | None = None,
    ) -> RunHandle:
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
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path.
            parent_dir: Where characterization data goes. Defaults to
                'Autofocus Characterization' under the live folder, where
                the button already puts it. Unused when no data is saved:
                that run writes nowhere, so where it would have written
                cannot refuse it.
            callbacks: Optional dict of callback functions.
            claim: A claim the caller holds -- the one
                ``session.diagnostic_claim()`` yields -- to run under
                instead of taking the scope. The run acts inside the
                caller's activity and cannot release its claim; it is
                refused if that claim no longer holds. None takes the scope
                for this run alone.
            run_trigger_source: Who asked for the run, recorded on it and
                named in refusals. The Autofocus button passes its own; a
                script keeps the default. Either way the run is attended
                -- its failures are shown -- unless it runs under *claim*.
            engineering_mode: Whether the run follows engineering-mode
                behaviour. None takes the session's; the GUI passes its live
                flag, which its plugin can change after the session exists.

        Returns:
            The committed run's handle, to wait on or to ignore;
            wait(timeout_s=...) on it gives the run's outcome.

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

        # Resolved only when data is saved, and then resolved rather than
        # left empty: the autofocus engine raises outright when asked to save
        # with nowhere to save to. Suppressing artifacts and delivering data
        # are not in conflict -- the run directory setup returns early on
        # suppression while the parent directory is still taken from the
        # plan. A run that saves nothing is given no directory at all, so
        # prepare()'s save-location gate never refuses it over a folder it
        # would never write to.
        if not save_characterization_data:
            parent_dir = None
        elif parent_dir is None:
            parent_dir = (
                pathlib.Path(settings['live_folder']).resolve() / 'Autofocus Characterization'
            )
        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            enable_image_saving=False,
            callbacks=callbacks,
            disable_saving_artifacts=True,
            save_autofocus_data=save_characterization_data,
            claim=claim,
            engineering_mode=engineering_mode,
        )

    def run_autofocus_all_steps(
        self,
        protocol: Protocol,
        callbacks: dict[str, typing.Callable] | None = None,
        run_trigger_source: str = 'api_autofocus_scan',
        engineering_mode: bool | None = None,
    ) -> RunHandle:
        """Autofocus at every step of *protocol*, and write the focus into it.

        One scan that visits each step with autofocus on, whatever each
        step's own autofocus setting, and captures nothing. When the scan
        completes, each step's Z becomes the focus found for it, written
        before the run lets go of the scope, so it is in the protocol when
        run_complete is sent.
        A scan that does not complete writes nothing, because it focused
        only some of the steps.

        The focus belongs to the steps it was found at, so a protocol whose
        steps changed during the scan -- a different number of them, or a
        step at a different position, channel or objective -- is left
        unchanged, and the person is told once ('Focus Not Saved'). The
        outcome's focus_written says which happened.

        Args:
            protocol: The protocol to focus. Its autofocus settings are not
                changed; only its Z values are written.
            callbacks: Optional dict of callback functions.
            run_trigger_source: Who asked for the run. The GUI's button
                passes its own; a script keeps the default.
            engineering_mode: Whether the run follows engineering-mode
                behaviour. None takes the session's; the GUI passes its live
                flag, which its plugin can change after the session exists.

        Returns:
            The committed run's handle, to wait on or to ignore;
            wait(timeout_s=...) on it gives the run's outcome.

        Raises:
            ProtocolRunRefusedError: The runner refused the request
                (already running, files still writing, hardware not
                connected, a step it cannot run); no state was committed.
        """
        scan = protocol.copy_for_execution()
        scan.modify_autofocus_all_steps(enabled=True)
        return self._run(
            protocol=scan,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name='af_scan',
            enable_image_saving=False,
            callbacks=callbacks,
            disable_saving_artifacts=True,
            engineering_mode=engineering_mode,
            write_focus_to=protocol,
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
    ) -> RunHandle:
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
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path.
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
            The committed run's handle, to wait on or to ignore;
            wait(timeout_s=...) on it gives the run's outcome.

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
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'Manual' / 'Z-Stacks'

        return self._run(
            protocol=protocol,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            run_trigger_source=run_trigger_source,
            max_scans=1,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            enable_image_saving=enable_image_saving,
            callbacks=callbacks,
            return_to_position=position if return_to_start else None,
            engineering_mode=engineering_mode,
        )

    def run_composite(
        self,
        sequence_name: str = 'composite',
        parent_dir: pathlib.Path | str | None = None,
        callbacks: dict[str, typing.Callable] | None = None,
        merge_timeout_s: float = 900.0,
        engineering_mode: bool | None = None,
    ) -> RunOutcome:
        """Capture one frame per acquiring channel and merge them.

        A composite is a single-position run through the same engine as
        every other run kind: one step per channel at the current stage
        position, each at its own stored focus, followed by the merge that
        combines them into one image.

        The channel set, the capture format and the depth all come from the
        settings snapshot rather than from any widget, so a headless caller
        gets the same run a GUI click does.

        The merged artifact is the run's real product, so this BLOCKS until
        the merge settles and returns the run's outcome, the artifact's path
        on it. A run that reported 'completed' while the merged file was
        missing would be indistinguishable from a successful one to every
        headless caller, which is the boundary this run kind exists to fix;
        and a composite merged from fewer channels than were asked for says
        so, with status 'incomplete' and the failed channels in
        ``captures``, beside its path.

        Args:
            sequence_name: The name of the protocol file the run saves in
                its folder ('.tsv' added): a name, not a path.
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
            The run's RunOutcome: ``merged`` True, ``artifact_path`` the
            merged composite, ``status`` 'completed' or 'incomplete', and
            ``captures`` what each channel produced.

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
        return settled

    def _run(
        self,
        protocol: Protocol,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        sequence_name: str,
        parent_dir: pathlib.Path | str | None = None,
        enable_image_saving: bool = True,
        callbacks: dict[str, typing.Callable] | None = None,
        return_to_position: dict | None = None,
        composite_thresholds_percent: dict | None = None,
        engineering_mode: bool | None = None,
        disable_saving_artifacts: bool = False,
        save_autofocus_data: bool = False,
        claim: HeldClaim | None = None,
        write_focus_to: Protocol | None = None,
    ) -> RunHandle:
        """Internal: configure and launch the sequenced capture executor.

        Returns:
            The committed run's handle.

        Raises:
            ProtocolRunRefusedError: The runner refused the request (already
                running, files still writing, empty/invalid protocol,
                hardware not connected); no state was committed and the
                user was already notified once.
        """
        # One copy of the settings, taken under the lock, for every value the
        # run reads from them: a caller on the worker pool (the GUI's Run, a
        # REST handler) reaches here while another thread may be writing the
        # store, and a run reading the live dict could take half an edit. The
        # autofocus snapshot below is the exception: its restorer writes back
        # into the live store, so it takes the lock itself.
        settings = self.session.get_settings_snapshot()

        import modules.config_helpers as config_helpers

        # The image mode and formats are the store's, read once here for every
        # run kind, so a script and the GUI's Run get the same files from the
        # same settings; session.set_image_mode is how a caller chooses. A
        # composite's merge reads its inputs back as 8-bit, so its rule is the
        # composite one.
        if run_mode == SequencedCaptureRunMode.SINGLE_COMPOSITE:
            image_capture_config = config_helpers.get_composite_image_capture_config_from_settings(
                settings
            )
        else:
            image_capture_config = config_helpers.get_image_capture_config_from_settings(settings)

        # A run that saves no artifacts and was given no directory writes
        # nowhere, and keeps None: prepare() reads it that way and does not
        # ask whether a folder it will never write to is usable.
        if parent_dir is None:
            if not disable_saving_artifacts:
                parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'ProtocolData'
        else:
            parent_dir = pathlib.Path(parent_dir)

        # One self-describing record per scan: the per-frame save path runs
        # thousands of times per session and cannot log its depth at info
        # level, so a scan's pixel format / on-disk encoding is otherwise
        # recoverable only by inspecting the output file tags afterward. This
        # line lets a support bundle state the mode the scan ran in and the
        # format the camera is delivering -- the mode's own depth is only
        # what it asked for, and an 8-bit camera delivers 8 in every mode.
        logger.info(
            f'[Protocol] scan "{sequence_name}" '
            f'image_mode={image_capture_config.image_mode} '
            f'pixel_format={self.session.scope.imaging.pixel_format_cached} '
            f'save_encoding={image_capture_config.save_encoding}'
        )

        autogain_settings = config_helpers.get_auto_gain_settings(settings)

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
            composite_thresholds_percent=composite_thresholds_percent,
            engineering_mode=engineering_mode,
            # Forwarded with the boundary's own names and its own defaults,
            # so this helper and the prepare it wraps stay one-to-one. No
            # existing caller passes either; both were reachable only from
            # inside the engine until a run kind needed to ask for them.
            disable_saving_artifacts=disable_saving_artifacts,
            save_autofocus_data=save_autofocus_data,
            write_focus_to=write_focus_to,
            borrowed_claim=claim.lend() if claim is not None else None,
            autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                self.session.settings, self.session.settings_lock
            ),
            **config_helpers.get_sequenced_run_settings(settings, run_mode=run_mode),
        )

        # Run-state truth is the session claim, committed inside
        # start()'s gate-and-commit -- a refusal means no state changed.
        return self._executor.start(plan)
