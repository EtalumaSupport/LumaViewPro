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
    """

    def __init__(self, reason: str, title: str, message: str):
        super().__init__(f'{reason}: {message}')
        self.reason = reason
        self.title = title
        self.message = message


class AutofocusAborted(Exception):  # noqa: N818 -- cancellation/abort signal, not an error; non-Error suffix is intentional
    """Autofocus run aborted by caller (e.g. user cancelled, protocol
    aborted, or app teardown)."""

    pass
