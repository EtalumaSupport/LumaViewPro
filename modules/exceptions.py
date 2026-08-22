# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Module/API-layer exception classes (raised from modules/, caught at UI).

For driver-layer hardware exceptions (HardwareError), see drivers/exceptions.py.
"""


class ProtocolError(Exception):
    """Protocol file parsing, validation, or execution error."""

    pass


class ConfigError(Exception):
    """Application configuration or settings error."""

    pass


class CaptureError(Exception):
    """Image capture, save, or processing failure."""

    pass


class ProtocolRunRefusedError(ProtocolError):
    """A sequenced run was refused before any state was committed.

    Raised by SequencedCaptureRunner.prepare() when a run cannot start
    (already running, files still writing, empty protocol, validation
    errors, hardware not connected). The refusal has already been logged
    and notified to the user when this is raised, so callers reconcile
    their own state without re-notifying.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: The notification title already shown to the user.
        message: The notification body already shown to the user.
        holder: The exclusive-activity claim owner at refusal time
            ('protocol' or 'recording'), or None when the refusal is
            not claim-shaped (validation, hardware, file drain).
        holder_trigger: Busy-with-what for run-shaped holders: the
            holding (or, for a file-drain refusal, the just-finished)
            run's run_trigger_source. None when the holder is not a
            run -- a recording has no trigger; its kind IS the holder.
    """

    def __init__(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
    ):
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger


class RecordingRefusedError(CaptureError):
    """A video recording start was refused before any state was committed.

    Raised when a recording cannot begin: by VideoRecordingEngine.start()
    when an exclusive activity -- a protocol run or another recording --
    already holds the session's activity claim or the engine is still
    draining, and by the recording controllers for the caller-shaped
    refusals they own (a previous recording still finishing, an inactive
    camera, an unknown exposure, insufficient disk). Mirrors the
    ProtocolRunRefusedError shape so callers reconcile state the same way
    in both directions.

    Attributes:
        reason: Machine-readable refusal code for callers that map
            refusals to responses (REST status codes, UI branches).
        title: Short user-facing refusal title.
        message: One-sentence user-facing refusal body.
        holder: The exclusive-activity claim owner at refusal time, or
            None when the refusal is not claim-shaped.
        holder_trigger: The holding run's run_trigger_source when the
            holder is 'protocol'; a recording holder has no trigger.
    """

    def __init__(
        self,
        reason: str,
        title: str,
        message: str,
        holder: 'str | None' = None,
        holder_trigger: 'str | None' = None,
    ):
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = title
        self.message = message
        self.holder = holder
        self.holder_trigger = holder_trigger


class HardwareCommandRefusedError(Exception):
    """A hardware command was refused: an exclusive activity holds the executor.

    Raised by the public hardware members (LED, camera and motion commands)
    when the executor that would carry the work will not accept it -- because
    a protocol run fenced it, or because the run disabled it outright. Both
    executor states make ``put()`` return None, and the caller cannot tell
    which one applies; asking whether work is accepted covers both, while
    asking why would need a list of reasons kept in sync with the executor.

    Distinct from the run and recording refusals, which are raised when an
    ACTIVITY is refused at start and which carry the title and body already
    shown to the user. This refusal reaches an external API caller that no
    notification path serves, so it carries no user-facing strings -- the
    caller that provoked it owns the response. Without it the command would
    be dropped silently, which is how a fenced write reaches no hardware and
    reports success.

    Attributes:
        reason: Machine-readable refusal code for callers that map refusals
            to responses (REST status codes, SDK branches).
        member: The public member that was refused, for the log and message.
    """

    def __init__(self, reason: str, member: str):
        super().__init__(f'{member} refused: {reason}')
        self.reason = reason
        self.member = member


class AutofocusAborted(Exception):  # noqa: N818 -- cancellation/abort signal, not an error; non-Error suffix is intentional
    """Autofocus run aborted by caller (e.g. user cancelled, protocol
    aborted, or app teardown)."""

    pass


class CameraSettingRejected(Exception):  # noqa: N818 -- named for the event it signals; one type covers the defect class
    """The camera driver rejected a state-changing setting apply.

    Raised by ImagingAPI setters (frame size, binning, pixel format) when
    a LIVE driver refuses the apply -- distinct from the camera-absent
    no-op, which stays a quiet sentinel per the missing-hardware contract.
    Success is observed by receiving the applied/delivered value, failure
    by this raise, so a caller cannot record a rejected apply as applied
    by forgetting to check a return code. The rejection has already been
    logged and notified to the user when this is raised.

    Attributes:
        setting: Machine-readable setting name (e.g. 'frame_size').
        requested: The value the caller asked for.
    """

    def __init__(self, setting: str, requested):
        super().__init__(f'{setting}: driver rejected {requested!r}')
        self.setting = setting
        self.requested = requested


class FrameDepthError(Exception):
    """A frame carries a payload value above its declared significant-bits depth.

    The downconvert to 8-bit scales against ``significant_bits`` -- the meaningful
    payload range the frame was captured under. A pixel larger than that range is
    a depth-contract violation: the significant-bits value is wrong for the data.
    Raised explicitly so the failure is loud and typed regardless of the
    downconvert arithmetic underneath -- a value-indexed LUT happens to raise
    IndexError today, but a scale-and-clip converter would silently map the
    over-range frame to white instead.

    Attributes:
        value: The offending pixel value.
        significant_bits: The declared depth it exceeded.
    """

    def __init__(self, value: int, significant_bits: int):
        super().__init__(
            f'frame value {value} exceeds the declared {significant_bits}-bit depth '
            f'(max {(1 << significant_bits) - 1}); the significant-bits contract is wrong'
        )
        self.value = value
        self.significant_bits = significant_bits
