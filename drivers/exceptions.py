# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Driver-layer exception classes (raised from drivers/, caught at module/API)."""

from collections.abc import Mapping
from typing import ClassVar

from lib.lumascope_error import Fault, _Umbrella
from lib.stage_interlock import INTERLOCK_TITLES


class HardwareError(_Umbrella, Exception):
    """A fault in the hardware (motor, LED, camera): the subject, never raised itself.

    ``except HardwareError`` catches every one; the child raised says which
    fault it was, with its own reason and title.
    """


class ConfigReadError(Fault, HardwareError):
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


class MotionInterlockError(Fault, HardwareError):
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

    _TITLES: ClassVar[Mapping[str, str]] = INTERLOCK_TITLES

    # Its words are the driver's record of the refusal; the person reads the
    # API's refusal, which says which interlock is open.
    words_for_person = False

    def __init__(self, reason: str, *, moved: bool, stopped: bool):
        super().__init__(f'motion interlock open: {reason} (moved={moved}, stopped={stopped})')
        self.reason = reason
        self.title = self._TITLES[reason]
        self.moved = moved
        self.stopped = stopped


class BoardNoReplyError(Fault, HardwareError):
    """A controller board did not answer a command within its timeout.

    The board may be busy, unpowered or unplugged; the same command may be
    answered when asked again. A motion command that got no answer may or
    may not have been taken, and the raise site's words say which.
    """

    reason = 'board_no_reply'
    title = 'Board Did Not Answer'

    words_for_person = False


class BoardReplyUnreadableError(Fault, HardwareError):
    """A controller board answered, and the answer could not be read.

    A reply that is not the number or record asked for, or that answers
    another command: the board is talking, and what it said cannot be used.
    """

    reason = 'board_reply_unreadable'
    title = 'Board Reply Unreadable'

    words_for_person = False


class BoardReportedError(Fault, HardwareError):
    """A controller board answered that it could not do what it was asked.

    The firmware's own error, a refused status, or a handshake it did not
    complete: the board heard the command and said no.
    """

    reason = 'board_reported_error'
    title = 'Board Reported an Error'

    words_for_person = False


class MotionNotCompletedError(Fault, HardwareError):
    """A driven motion did not reach its end.

    An axis that did not home, arrive or settle in the time it is given, or
    a home that ended before its phases did. Whether a Stop ended it is the
    API's stop generation to say, never this type.
    """

    reason = 'motion_not_completed'
    title = 'Motion Did Not Complete'

    words_for_person = False


class BoardPortFailedError(Fault, HardwareError):
    """The serial port to a controller board failed, or is not open.

    A cable or power state: nothing reached the board.
    """

    reason = 'board_port_failed'
    title = 'Board Connection Failed'

    words_for_person = False


class LedFirmwareWedgedError(Fault, HardwareError):
    """The LED controller's firmware is stuck and did not recover on a soft reset.

    Only a power cycle brings it back, and its words say so.
    """

    reason = 'led_firmware_wedged'
    title = 'LED Controller Stuck'

    words_for_person = True


class CameraReadFailedError(Fault, HardwareError):
    """The camera did not answer a read of one of its values, or a probe of one.

    The words carry the SDK's own text, so the person reads the title.
    """

    reason = 'camera_read_failed'
    title = 'Camera Read Failed'

    words_for_person = False


class CameraWriteFailedError(Fault, HardwareError):
    """The camera refused or did not take a write of one of its settings.

    The words carry the SDK's own text, so the person reads the title.
    """

    reason = 'camera_write_failed'
    title = 'Camera Write Failed'

    words_for_person = False


class CameraNoAnswerError(Fault, HardwareError):
    """The camera did not answer a write of one of its settings in time.

    An SDK or USB timeout on the access that decides whether the write
    happened. The camera is still on the bus and may answer the same write
    sent again; whether this one was applied is not known.
    """

    reason = 'camera_no_answer'
    title = 'Camera Did Not Answer'

    words_for_person = False


class CameraUsbNotFoundError(Fault, HardwareError):
    """No USB device the camera driver can stream from answers to its identity.

    On Windows: none bound to WinUSB, which is what the stream opens.
    The words carry the identity and the system's own text.
    """

    reason = 'camera_usb_not_found'
    title = 'Camera Not Found on USB'

    words_for_person = False


class CameraUsbLookupFailedError(Fault, HardwareError):
    """The system's lookup of the camera's USB device itself failed.

    Not an absence: the lookup did not complete, so whether the device is
    there is not known. The words name the call and its error code.
    """

    reason = 'camera_usb_lookup_failed'
    title = 'Camera USB Lookup Failed'

    words_for_person = False


class CameraUsbDriverUnusableError(Fault, HardwareError):
    """The camera's USB device is bound to a driver LumaViewPro cannot use.

    Windows binds each of the Lumascope Classic's two USB identities to a
    driver on its own; a device with none, or one the firmware upload or
    the stream cannot drive, cannot come up however it is cabled. The words
    name the device, the driver it is bound to and what it needs.
    """

    reason = 'camera_usb_driver_unusable'
    title = 'Camera Driver Not Usable'

    words_for_person = False


class CameraUsbOpenFailedError(Fault, HardwareError):
    """The camera's USB device was found and could not be opened or set up
    for streaming. The words name the call and its error code.
    """

    reason = 'camera_usb_open_failed'
    title = 'Camera USB Open Failed'

    words_for_person = False


class CameraStreamStartFailedError(Fault, HardwareError):
    """The stream's own set-up on an opened camera failed. The words carry
    the library's text.
    """

    reason = 'camera_stream_start_failed'
    title = 'Camera Stream Did Not Start'

    words_for_person = False


class CameraFirmwareMissingError(Fault, Exception):
    """The firmware file the camera's driver uploads is not in the installation.

    The installation is at fault, not the hardware or its cable, so this is
    not a ``HardwareError``. The words name each place the driver looked.
    """

    reason = 'camera_firmware_missing'
    title = 'Camera Firmware File Missing'

    words_for_person = False

    def __init__(self, searched):
        super().__init__(f'camera firmware file not found; searched: {", ".join(searched)}')
        self.searched = tuple(searched)


class PartAbsentError(Fault, HardwareError):
    """A driver was asked for a part it does not have: an axis, a turret, a
    board, an active camera or an SDK node.

    A fault, not a refusal: the API refuses a command for a part the scope
    lacks before the driver is asked (``axis_absent``, ``not_on_this_scope``,
    ``not_connected``), so only a caller below the API reaches this.
    """

    reason = 'part_absent'
    title = 'Part Not Present'

    words_for_person = False
