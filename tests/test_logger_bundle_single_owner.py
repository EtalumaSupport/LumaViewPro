"""Regression: the bundle file handlers have a single owner (the root logger).

Bug shape (pre-fix): the same RotatingFileHandler instances that write the
shared bundle (lvpmain.log / errors.log) were attached to the root logger AND
to the LVP loggers (`lvp_logger`, `LVP`) AND to `kivy`. A record that
propagated up to root was then written to the bundle a SECOND time through
root's copy of the same handler. This was masked only by disabling propagation
in non-debug, so a debug run -- exactly when the bundle matters for diagnosis --
doubled every LVP line.

Fix: root is the single owner of the bundle handlers; every ordinary logger
reaches the bundle by propagating to root, and no logger holds its own copy.
The deliberately-separate firehose loggers (camera/protocol/serial/... write
their own .log and stay out of the main bundle via propagate=False + their own
handlers) are a different, intentional routing and are left alone.

Asserted from source: the live logger tree is global state that pytest's
logging capture and other tests mutate between runs, so runtime handler/
propagate checks are unstable (the existing root-logging tests assert on source
for the same reason).
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1] / 'lvp_logger.py').read_text(encoding='utf-8')


def _stmt(holder: str, tail: str) -> re.Pattern:
    # Anchor the holder at the start of a line so `logger` / `_lvp_parent` /
    # `kivy_logger` do not spuriously match the firehose loggers whose names
    # END in 'logger' (camera_logger, protocol_logger, serial_logger, ...).
    return re.compile(rf'^{re.escape(holder)}{re.escape(tail)}', re.MULTILINE)


def test_root_is_sole_owner_of_bundle_handlers():
    # Root owns them...
    assert '_root_logger.addHandler(file_handler)' in _SRC
    assert '_root_logger.addHandler(error_file_handler)' in _SRC
    # ...and the ordinary LVP loggers must NOT re-attach the same instances
    # (that duplicate attachment is what doubled the bundle on propagation).
    for holder in ('logger', '_lvp_parent', 'kivy_logger'):
        for handler in ('file_handler', 'error_file_handler'):
            assert not _stmt(holder, f'.addHandler({handler})').search(_SRC), (
                f'{holder} re-attaches {handler}, which the root logger already '
                f'owns; a propagated record would write the bundle twice. It '
                f'must reach the bundle by propagation to root, not its own copy.'
            )


def test_ordinary_lvp_loggers_reach_bundle_by_propagation():
    # propagate=False on these was the mask that hid the double-write; it must
    # not come back (the firehose loggers keep their own propagate=False -- a
    # different, intentional routing -- so target these three by name).
    for holder in ('logger', '_lvp_parent', 'kivy_logger'):
        assert not _stmt(holder, '.propagate = False').search(_SRC), (
            f'{holder}.propagate must stay True so it reaches the single bundle '
            f'owner (root) by propagation; propagate=False was the double-write '
            f'mask.'
        )
