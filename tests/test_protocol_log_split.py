# Copyright Etaluma, Inc.
"""Regression test: protocol-execution detail routes to a dedicated log.

A long protocol soak emitted tens of thousands of per-step lines (step
records, per-channel LED, image-captured events) into the main log,
swamping it. protocol_image_writer now logs through a dedicated
``LVP.protocol`` logger (propagate=False -> protocol.log), mirroring the
serial / camera / metrics split, so the main log stays readable while the
full run history is preserved.

These are source-scan tests because conftest mocks lvp_logger wholesale
during pytest, so the real logger configuration never executes at runtime.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _src(rel):
    return (REPO_ROOT / rel).read_text()


def test_protocol_logger_defined_non_propagating():
    """lvp_logger must define a dedicated LVP.protocol logger that writes
    to protocol.log and does not propagate into the main log."""
    # pin-justified: conftest mocks lvp_logger wholesale, so the real
    # logger configuration never executes in pytest (see module docstring).
    src = _src('lvp_logger.py')
    assert "PROTOCOL_LOG_FILE" in src and "'protocol.log'" in src, (
        'lvp_logger must define PROTOCOL_LOG_FILE pointing at protocol.log'
    )
    idx = src.find("getLogger('LVP.protocol')")
    assert idx != -1, "lvp_logger must create the 'LVP.protocol' logger"
    window = src[idx : idx + 400]
    assert 'propagate = False' in window, (
        'the LVP.protocol logger must set propagate = False to keep '
        'protocol detail out of the main log'
    )


def test_protocol_image_writer_uses_protocol_logger():
    """protocol_image_writer -- the per-step narrative bulk -- must log
    through the dedicated protocol logger, not the root logger."""
    # pin-justified: import-binding contract; conftest mocks lvp_logger
    # wholesale, so the binding cannot be observed at runtime in pytest.
    src = _src('modules/protocol_image_writer.py')
    assert 'from lvp_logger import protocol_logger as logger' in src, (
        'protocol_image_writer must bind its module logger to '
        'protocol_logger so its per-step detail lands in protocol.log'
    )
