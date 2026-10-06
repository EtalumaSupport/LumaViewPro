# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Protocol cleanup / shutdown logic.

Restores LED, autofocus, camera state and fires completion callbacks.
Extracted from ``sequenced_capture_runner.py`` during the
protocol-decomposition refactor.
"""

from __future__ import annotations

import pathlib
import threading
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING

from lvp_logger import logger

from modules.autofocus_runner import AF_DATA_WRITE_WAIT_S
from modules.exceptions import RunCleanupFailedError, SlowFileWritesNotice
from modules.lumascope_api.illumination import (
    LedTransition,
    LedTransitionCtx,
    resolve_end_state,
)
from modules.protocol_image_writer import SLOW_WRITE_BLOCKED_WARN_S
from modules.protocol_state_machine import ProtocolState
from modules.run_outcome import RunEnding

if TYPE_CHECKING:
    from collections.abc import Callable

    from modules.autofocus_thread import AutofocusThread
    from modules.config_helpers import AutofocusSnapshot
    from modules.lumascope_api import Lumascope
    from modules.protocol import Protocol
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_image_writer import RunWriteBatch


from modules.kivy_utils import schedule_ui as _schedule_ui


def _schedule_cleanup_ui(
    func,
    step_label: str,
    cleanup_errors: list[tuple[str, str]],
    summary_sent: threading.Event,
) -> None:
    """Schedule a cleanup UI callback that must not take the app down.

    Every `try` in this module collects its failure into `cleanup_errors`
    and keeps going, because a run's data is already written by the time
    cleanup starts and no cleanup step is worth losing the session over.
    A callback handed to `schedule_ui` was outside that contract whenever
    it was genuinely deferred: it runs on a later Clock tick, so the `try`
    that scheduled it has already returned, and the app's crash guard
    re-raises anything it cannot pin on a plugin. A panel-sync failure
    therefore terminated LumaViewPro at the end of every protocol run.

    The re-raise is right as a DEFAULT -- a core bug should be loud -- and
    is left alone everywhere else. It is wrong for these callbacks for the
    same reason the code around them is fault-tolerant, and for the same
    reason a non-fatal failure during a run is log-only: an unattended run
    must not lose its application to a cosmetic restore step.

    Which channel reports the failure depends on WHEN it happens, because
    the summary is emitted once, partway through:

    - before the summary (the callback ran inline -- headless, REST, or a
      test dispatcher): collected into `cleanup_errors` exactly as if the
      surrounding `try` had caught it, so the one summary still carries
      every failed step and its count stays honest.
    - after it (the GUI case, a real Clock tick): the summary has already
      gone, so the callback reports itself.

    `step_label` is the same wording the surrounding `except` blocks use,
    so a step reads identically whichever channel carried it.
    """

    def _guarded(dt):
        try:
            return func(dt)
        except Exception as ex:
            if not summary_sent.is_set():
                # The summary carries the step and its message; the
                # traceback belongs in the log that ships with a bundle.
                logger.exception(f'[PROTOCOL] {step_label} failed after the run')
                cleanup_errors.append((step_label, f'{type(ex).__name__}: {ex}'))
                return
            from modules.notification_center import notifications

            # Says nothing of the images: this runs for any step after the
            # run, whether or not its files were all written, and the files
            # are told on their own. Chained, so the one report logs the
            # traceback.
            failed = RunCleanupFailedError([(step_label, f'{type(ex).__name__}: {ex}')])
            failed.__cause__ = ex
            notifications.report_outcome(failed, solicited=False, category='Protocol')

    _schedule_ui(_guarded, 0)


class RunCompleteNotice:
    """The run's one ``run_complete``, carrying the run's values by value.

    Sent by the run's end once the run has let go of the scope -- on every
    path, a cleanup that raised included -- and before the run's files are
    handed their completion, so every subscriber hears ``run_complete``
    once, when it can act on the scope, and before ``files_complete``.
    Built while the run's fields are still its own: a successor started
    after the release replaces them.
    """

    def __init__(
        self,
        callbacks: ProtocolCallbacks,
        *,
        protocol: Protocol,
        ending: RunEnding,
        run_dir: pathlib.Path | None,
    ):
        self._callbacks = callbacks
        self._protocol = protocol
        self._ending = ending
        self._run_dir = run_dir

    @property
    def ending(self) -> RunEnding:
        """The ending this notice carries to the run's subscribers."""
        return self._ending

    def send(self) -> None:
        """Schedule ``run_complete``; a failure inside it is reported by the callback's own guard.

        Sent after the run's cleanup summary, so the failure is its own report.
        """
        if not self._callbacks.run_complete:
            return
        summary_sent = threading.Event()
        summary_sent.set()
        _schedule_cleanup_ui(
            lambda dt: self._callbacks.run_complete(
                protocol=self._protocol,
                status=self._ending.status,
                ending=self._ending,
                run_dir=self._run_dir,
            ),
            'Run-complete callback',
            [],
            summary_sent,
        )


def schedule_files_complete(
    callbacks: ProtocolCallbacks,
    *,
    protocol: Protocol,
    run_dir: pathlib.Path | None,
    files: str,
) -> None:
    """Schedule the run's one ``files_complete``, after its cleanup summary.

    ``files`` is the run's write outcome: ``'written'``, or ``'incomplete'``
    when some of its images are not on disk.
    """
    if not callbacks.files_complete:
        return
    summary_sent = threading.Event()
    summary_sent.set()
    _schedule_cleanup_ui(
        lambda dt: callbacks.files_complete(protocol=protocol, run_dir=run_dir, files=files),
        'Files-complete callback',
        [],
        summary_sent,
    )


# How long cleanup waits for an in-flight autofocus to unwind: its restore,
# plus the wait for its data write that ends it.
_AF_UNWIND_WAIT_S = 5.0 + AF_DATA_WRITE_WAIT_S


def run_cleanup(
    *,
    # State
    get_state_fn: Callable[[], ProtocolState],
    set_state_fn: Callable[[ProtocolState], None],
    scan_in_progress: threading.Event,
    # True when the run died on a fault that force-darkened the sample
    # (stalled writer, dead camera, disk floor) rather than finishing or
    # being stopped by the user. Required, not defaulted: every caller
    # states which kind of end this is. Read once by the caller, so the
    # decision cannot flip mid-cleanup.
    forced_dark: bool,
    # Saved original states
    leds_state_at_end: str,
    original_led_states: dict,
    autofocus_snapshot: AutofocusSnapshot,
    saved_camera_state: dict,
    return_to_position: dict | None,
    # Dependencies
    scope: Lumascope,
    callbacks: ProtocolCallbacks,
    # Executor functions
    apply_led_transition_fn: Callable[[LedTransition, LedTransitionCtx], object],
    default_move_fn: Callable[..., object],
    cancel_scheduled_events_fn: Callable[[], None],
    # IO executors; the IO and CAMERA lanes are the scope's
    autofocus_thread: AutofocusThread | None,
    # THIS run's writes: read for the run-end summary. Its completion --
    # the record's, and files_complete -- is the batch's, once the last
    # write lands.
    write_batch: RunWriteBatch,
    logger_name: str = 'SequencedCaptureRunner',
    # How the run ended, and why.
    ending: RunEnding,
    # Where the run's outcome takes the names of the steps that failed to
    # put the scope back; called once, with none when every step finished.
    record_cleanup_failures: Callable[[tuple[str, ...]], None],
) -> bool:
    """Core cleanup logic -- restores state, ends executors.

    Called from ``SequencedCaptureRunner._cleanup_inner()``. ending is
    required so the cleanup site states the run's true terminal outcome.
    The run's writes are not ended here: they are the run's batch's, which
    the caller closes on every path out, and which writes every image the
    run captured however the run ended. Nor is run_complete sent here: the
    run sends it once it has let go of the scope, so a subscriber can act
    on the scope when told.

    Returns True when the RUN_END LED transition actually applied -- the
    run's LED end-state is decided. False (or a raise anywhere in here)
    means it is NOT decided, and the caller must darken before releasing
    the lease: a lit channel with no owner to turn it off cooks the
    sample, and the lease release itself deliberately leaves LEDs as-is.
    """
    # Transition to COMPLETING (or stay in ERROR if that's how we got here)
    if get_state_fn() not in (ProtocolState.COMPLETING, ProtocolState.ERROR, ProtocolState.IDLE):
        set_state_fn(ProtocolState.COMPLETING)

    # Cleanup runs because abort already fired (or the run is finishing
    # naturally). The abort signal -- now owned by protocol_thread -- is
    # already set if this is an abort; setting it again here would be
    # redundant and was dropped in B3.
    scan_in_progress.clear()

    # Collect cleanup-step failures so a single summary notification at
    # the end tells the user what went wrong. Each except continues to
    # the next step (fault tolerance -- all six must run regardless of
    # any one failing); total silence at the end was the bug. One
    # summary popup, not six.
    cleanup_errors: list[tuple[str, str]] = []
    # Flipped once the summary below has gone out. A guarded UI callback
    # that fails before this is collected into the summary like every
    # other step; one that fails after it has to report itself.
    summary_sent = threading.Event()

    try:
        cancel_scheduled_events_fn()
    except Exception as ex:
        cleanup_errors.append(('Cancel scheduled events', f'{type(ex).__name__}: {ex}'))

    # --- Unwind any in-flight autofocus BEFORE restoring LEDs ---
    # The AF worker lights its own channel during setup and restores
    # LED / camera / Z state in its finally block. If the LED restore
    # below ran first, a still-unwinding AF run would re-light or
    # re-restore on top of it, and the protocol's intended end state
    # would lose the race -- worst case an AF LED left on overnight.
    # The AF Future resolves only after that finally chain finishes,
    # so waiting on it (bounded, so a wedged AF run cannot block
    # cleanup) guarantees the LED restore below runs last. The chain
    # ends by waiting for the sweep's data write, which an aborted run
    # still makes, so the bound covers that wait on top of the unwind.
    # A run that failed during start() never dispatched anything, so a live
    # AF future here belongs to SOMEONE ELSE -- most likely the very holder
    # whose lease refusal failed this run. Aborting it would steal the
    # operation the refusal deferred to.
    if autofocus_thread is not None and ending.status != 'failed_at_start':
        _af_future = autofocus_thread.current_future
        if _af_future is not None and not _af_future.done():
            autofocus_thread.abort()
            try:
                # Returns the run's exception (normally AutofocusAborted)
                # without raising it; raises TimeoutError on the bound.
                _af_future.exception(timeout=_AF_UNWIND_WAIT_S)
            except TimeoutError:
                logger.warning(
                    f'[{logger_name}] Cleanup: autofocus still unwinding '
                    f'after {_AF_UNWIND_WAIT_S:.1f} s; its exit path restores '
                    'LED/camera state when it finishes'
                )
            except Exception as ex:
                logger.warning(
                    f'[{logger_name}] Cleanup: error waiting for autofocus '
                    f'to unwind: {type(ex).__name__}: {ex}'
                )

    # --- Restore LEDs ---
    # One authority diff sets the run's end-state: off, or back to the
    # channels that were lit before the run. The diff turns off any channel
    # not in the target set, then asserts the target -- so restoring more
    # than one pre-run channel does not flash, where the old per-channel
    # restore loop extinguished each channel as it lit the next. An empty
    # restore target (nothing was lit pre-run) collapses to off by
    # construction. apply(RUN_END) runs on the still-held lease and serializes
    # on the protocol IO queue, so the end-state off cannot race the
    # return-to-position move across the shared serial bus.
    led_end_state_applied = False
    # A run that ended before it took the camera (a Stop, or a stuck lane,
    # during the wait for the lane) changed no LED and holds no snapshot,
    # so there is nothing to return to: the resolver would read the missing
    # snapshot as "nothing was lit" and darken a sample a still may be
    # grabbing under right now.
    if original_led_states is None:
        logger.info(f'[{logger_name}] Cleanup: the run never took the camera; LEDs left as found')
        # Decided, not undecided: "as found" is the end state, and the
        # caller's fallback for an undecided one is to darken.
        led_end_state_applied = True
    else:
        try:
            # A fatal abort's terminal LED state is DARK regardless of the user's
            # end policy: force_off already darkened the sample at the fault
            # site, and this forced-OFF RUN_END re-asserts dark against any step
            # that raced the abort and re-lit a channel (the OFF diff serializes
            # after such a re-light on the same FIFO protocol queue, so off
            # wins). Asserting OFF -- not skipping the restore -- is the point: a
            # skipped restore would leave a raced re-light on forever. User Stop
            # keeps the configured policy.
            end_policy, snapshot_lit = resolve_end_state(
                'off' if forced_dark else leds_state_at_end,
                original_led_states,
                scope.illumination.state_color2ch,
            )
            if end_policy is None:
                logger.error(f'Unsupported LEDs state at end value: {leds_state_at_end}')
            else:
                apply_led_transition_fn(
                    LedTransition.RUN_END,
                    LedTransitionCtx(end_policy=end_policy, snapshot_lit=snapshot_lit),
                )
                led_end_state_applied = True
        except CancelledError:
            # The protocol queue was cleared and this restore task cancelled
            # before it ran. A superseding run/abort cycle is one canceller --
            # but so are executor shutdown, an unwedge/quarantine, and the
            # end-of-protocol-mode drain, and none of those re-asserts LED
            # state. So the end-state stays undecided and the caller darkens;
            # a superseding run re-lights per step, costing at most a
            # transient dark blip during rapid run cycling. Not surfaced as a
            # failure -- doing so produced a popup per cycle when the run
            # button was clicked rapidly.
            logger.info(
                f'[{logger_name}] Cleanup: LED restore superseded by an overlapping run/abort cycle'
            )
        except Exception as ex:
            cleanup_errors.append(('Restore LED states', f'{type(ex).__name__}: {ex}'))
    logger.info(f'[{logger_name}] Cleanup: LED restore complete')

    # --- Restore layer shader / false-color (UI side) ---
    # Each protocol step calls layer_control.apply_settings() which
    # writes the OpenGL shader white_point for that layer's
    # false-color (Red tint for the Red step, Green tint for Green,
    # etc.). Without this restore the last step's shader stays
    # active and tints the live preview after protocol stop. Cluster
    # sibling of LED-state-hygiene-at-transition (#666 / #659 /
    # #617): driver LED state was already cleared above; this is
    # the sibling UI-shader-state clear. Bugs cluster -- one cleanup
    # pass covers both halves.
    try:
        if callbacks.restore_layer_shader:
            _schedule_cleanup_ui(
                lambda dt: callbacks.restore_layer_shader(),
                'Restore layer shader',
                cleanup_errors,
                summary_sent,
            )
    except Exception as ex:
        cleanup_errors.append(('Restore layer shader', f'{type(ex).__name__}: {ex}'))

    # --- Restore autofocus states ---
    # Empty states (the common case when no AF was active for this scan)
    # skip the restore with one debug line. Iterating an absent snapshot
    # once fired ERROR every scan, burying real failure signal under
    # thousands of spurious lines; the snapshot is required now, so only
    # the empty case remains. The restorer rides the snapshot: the run
    # writes back to the same dict it was read from, whichever process
    # owns it.
    if not autofocus_snapshot.states:
        logger.debug('[PROTOCOL] No autofocus states to restore')
    else:
        try:
            for layer, layer_data in autofocus_snapshot.states.items():
                autofocus_snapshot.restore(layer=layer, value=layer_data)
        except Exception as ex:
            cleanup_errors.append(('Restore autofocus states', f'{type(ex).__name__}: {ex}'))

    # --- Put the layer panel back on the settings ---
    # The run displayed each step in the panel without writing the user's
    # settings; the panel now shows the settings again. Once per run,
    # outside the restore above: it does not depend on any autofocus state
    # having been snapshotted.
    try:
        if callbacks.sync_layer_widgets:
            _schedule_cleanup_ui(
                lambda dt: callbacks.sync_layer_widgets(),
                'Sync layer panel',
                cleanup_errors,
                summary_sent,
            )
    except Exception as ex:
        cleanup_errors.append(('Sync layer panel', f'{type(ex).__name__}: {ex}'))

    # --- Restore camera gain and exposure ---
    # Before the return moves and the executors' end: live preview after the
    # stop must not briefly run at the run's gain and exposure. The restore
    # dispatches onto the camera lane; cleanup acts under the run's taking,
    # so it goes through the run's door while the lane is still in protocol
    # mode.
    try:
        if saved_camera_state:
            tag = saved_camera_state.get('tag', '?')
            logger.info(f'[{logger_name}] Cleanup: restoring camera state tag={tag}')
            scope.imaging.restore_camera_state(saved_camera_state)
    except Exception as ex:
        cleanup_errors.append(('Restore camera gain/exposure', f'{type(ex).__name__}: {ex}'))

    # --- Return to position ---
    try:
        if return_to_position is not None:
            logger.info(
                f'[{logger_name}] Cleanup: returning to position '
                f'x={return_to_position["x"]}, y={return_to_position["y"]}, z={return_to_position["z"]}'
            )
            default_move_fn(
                px=return_to_position['x'],
                py=return_to_position['y'],
                z=return_to_position['z'],
            )
            logger.info(f'[{logger_name}] Cleanup: return-to-position move issued')
    except CancelledError:
        # Same hand-off as the LED restore above: a superseding run/abort
        # cycle cancelled the queued move; the new cycle owns stage position.
        logger.info(
            f'[{logger_name}] Cleanup: return-to-position superseded by an '
            'overlapping run/abort cycle'
        )
    except Exception as ex:
        cleanup_errors.append(('Return to position', f'{type(ex).__name__}: {ex}'))

    # --- End executors ---
    scan_in_progress.clear()

    io_executor = scope.io_lane()
    camera_executor = scope.camera_lane()
    io_executor.protocol_end()
    # Wait for any task that was in-flight when protocol_end fired to
    # finish before we mutate scope / camera / settings state below --
    # an in-flight task on the io_executor worker may be reading the
    # same state. Bounded so a wedged task can't block cleanup
    # indefinitely; if the timeout fires we log and proceed.
    if not io_executor.wait_for_idle(timeout=2.0):
        logger.warning(
            f'[{logger_name}] Cleanup: io_executor still mid-task '
            'after 2.0 s wait; proceeding to teardown anyway'
        )
    if autofocus_thread is not None:
        # Signal any lingering AF run to unwind. abort() is a no-op when
        # the thread is idle, so this is always safe to call.
        autofocus_thread.abort()
    camera_executor.protocol_end()
    logger.info(f'[{logger_name}] Cleanup: protocol_end called on all executors')

    io_executor.clear_protocol_pending()
    camera_executor.clear_protocol_pending()
    # The run is NOT ended here. The phase returns to IDLE in the caller's
    # finally, after the activity claim is released -- one writer, on a
    # path that runs even when a step in here raises. Ending the run from
    # inside this function is what used to admit the next run while the
    # teardown was still handing resources back.

    # Surface a single summary if any cleanup step failed. Fault
    # tolerance ran each step regardless; the user needs to know LED
    # state, camera settings, or stage position may not be what they
    # expect.
    # The run's outcome names the steps, so a caller waiting on the run
    # learns the scope was not put back; the person is told once, here.
    record_cleanup_failures(tuple(step for step, _ in cleanup_errors))
    if cleanup_errors:
        from modules.notification_center import notifications

        notifications.report_outcome(
            RunCleanupFailedError(cleanup_errors), solicited=False, category='Protocol'
        )

    # The one summary has now gone out (or there was nothing to say). Any
    # guarded UI callback that fails from here on has missed it and must
    # report itself instead of appending where nobody will read.
    summary_sent.set()

    # Sustained-slow-write warning, demand-relative: the time this run's
    # capture loop spent blocked waiting for a write slot. An absolute MB/s
    # floor false-fires on healthy machines (PERFORMANCE_BUDGETS.md
    # protocol_write_backpressure_wait_s), so the trigger is the run's own
    # unmet demand. Surfaced at run end because mid-run non-fatal popups are
    # suppressed; the first crossing already logged from the run's write batch.
    blocked_s = write_batch.blocked_s
    if blocked_s >= SLOW_WRITE_BLOCKED_WARN_S:
        from modules.notification_center import notifications

        notifications.report_outcome(SlowFileWritesNotice(), solicited=False, category='Protocol')

    # run_complete and files_complete are the run's to send, once it has let
    # go of the scope.
    logger.info(f'[{logger_name}] Run ended: status={ending.status} reason={ending.reason}')
    logger.info(f'[{logger_name}] Cleanup: pending_writes={write_batch.pending}')

    # Map the footprint right after a protocol run. No-op unless the memory
    # profiler is enabled.
    from lib import memory_profile

    memory_profile.snapshot('post_protocol')

    return led_end_state_applied
