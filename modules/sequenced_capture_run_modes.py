
import enum

class SequencedCaptureRunMode(enum.Enum):
    FULL_PROTOCOL = 'full_protocol'
    SINGLE_SCAN = 'single_scan'
    SINGLE_ZSTACK = 'single_zstack'
    SINGLE_AUTOFOCUS_SCAN = 'single_autofocus_scan'
    SINGLE_AUTOFOCUS = 'single_autofocus'
