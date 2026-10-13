# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Driver-layer exception classes (raised from drivers/, caught at module/API)."""

from typing import ClassVar

from lib.lumascope_error import Fault


class HardwareError(Fault, Exception):
    """Hardware communication or configuration failure (motor, LED, camera)."""

    reason = 'hardware_error'
    title = 'Hardware Error'

    words_for_person = True


class ConfigReadError(HardwareError):
    """The board's per-unit config could not be READ (no answer, or an
    unparseable payload), or a value in it is not one the field can hold
    (an optics value that is not a number).

    Distinct from a board that answers with an empty or minimal config:
    after a failed read the per-unit values may exist on the board but
    are unavailable, so consumers that would trust "no per-unit value
    present" need to know the difference or they silently serve another
    source's answer for a unit that has its own.
    """

    reason = 'config_unread'
    title = 'Board Config Not Read'

    words_for_person = True


class MotionInterlockError(HardwareError):
    """The board refused a motion command because a hardware interlock is open.

    Raised by a driver whose board guards motion with its own inputs (a
    lid switch, the stage's power), read on the wire with the command they
    guard. The API turns it into its refusal or its failed home, so it
    carries what the API needs to say which one happened: whether any part
    of the refused command had already moved, and whether the driver tried
    to stop the motors in refusing it.

    Attributes:
        reason: ``'lid_open'`` or ``'stage_unpowered'``.
        moved: Whether anything of the refused command had moved.
        stopped: Whether the driver sent every motor a stop in refusing it,
            settled or not; False when no stop was sent, as for a stage
            without power. The API counts a sent stop as a Stop.
    """

    _TITLES: ClassVar[dict[str, str]] = {
        'lid_open': 'Lid Open',
        'stage_unpowered': 'No Stage Power',
    }

    # Its words are the driver's record of the refusal; the person reads the
    # API's refusal, which says which interlock is open.
    words_for_person = False

    def __init__(self, reason: str, *, moved: bool, stopped: bool):
        super().__init__(f'motion interlock open: {reason} (moved={moved}, stopped={stopped})')
        self.reason = reason
        self.title = self._TITLES[reason]
        self.moved = moved
        self.stopped = stopped


class BoardNoReplyError(HardwareError):
    """A controller board did not answer a command within its timeout.

    The board may be busy, unpowered or unplugged; the same command may be
    answered when asked again. A motion command that got no answer may or
    may not have been taken, and the raise site's words say which.
    """

    reason = 'board_no_reply'
    title = 'Board Did Not Answer'

    words_for_person = False


class BoardReplyUnreadableError(HardwareError):
    """A controller board answered, and the answer could not be read.

    A reply that is not the number or record asked for, or that answers
    another command: the board is talking, and what it said cannot be used.
    """

    reason = 'board_reply_unreadable'
    title = 'Board Reply Unreadable'

    words_for_person = False


class BoardReportedError(HardwareError):
    """A controller board answered that it could not do what it was asked.

    The firmware's own error, a refused status, or a handshake it did not
    complete: the board heard the command and said no.
    """

    reason = 'board_reported_error'
    title = 'Board Reported an Error'

    words_for_person = False


class MotionNotCompletedError(HardwareError):
    """A driven motion did not reach its end.

    An axis that did not home, arrive or settle in the time it is given, or
    a home that ended before its phases did. Whether a Stop ended it is the
    API's stop generation to say, never this type.
    """

    reason = 'motion_not_completed'
    title = 'Motion Did Not Complete'

    words_for_person = False


class BoardPortFailedError(HardwareError):
    """The serial port to a controller board failed, or is not open.

    A cable or power state: nothing reached the board.
    """

    reason = 'board_port_failed'
    title = 'Board Connection Failed'

    words_for_person = False


class LedFirmwareWedgedError(HardwareError):
    """The LED controller's firmware is stuck and did not recover on a soft reset.

    Only a power cycle brings it back, and its words say so.
    """

    reason = 'led_firmware_wedged'
    title = 'LED Controller Stuck'

    words_for_person = True


class CameraReadFailedError(HardwareError):
    """The camera did not answer a read of one of its values, or a probe of one.

    The words carry the SDK's own text, so the person reads the title.
    """

    reason = 'camera_read_failed'
    title = 'Camera Read Failed'

    words_for_person = False


class CameraWriteFailedError(HardwareError):
    """The camera refused or did not take a write of one of its settings.

    The words carry the SDK's own text, so the person reads the title.
    """

    reason = 'camera_write_failed'
    title = 'Camera Write Failed'

    words_for_person = False


class PartAbsentError(HardwareError):
    """A driver was asked for a part it does not have: an axis, a turret, a
    board, an active camera or an SDK node.

    A fault, not a refusal: the API refuses a command for a part the scope
    lacks before the driver is asked (``axis_absent``, ``not_connected``),
    so only a caller below the API reaches this.
    """

    reason = 'part_absent'
    title = 'Part Not Present'

    words_for_person = False
