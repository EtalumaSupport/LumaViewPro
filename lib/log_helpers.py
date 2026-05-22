# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Logger composition helpers.

Pure functions with no Kivy / platformdirs / config dependencies, so
they can be imported (and tested) without dragging in the heavy
``lvp_logger`` module's startup. The ``log_to`` helper is the shared
dual / multi-write primitive that subsystem-mirroring callers use
(camera.log today; motor.log / led.log / ids.log when those land).
"""


def log_to(*loggers, level: str, message: str) -> None:
    """Mirror one message to multiple loggers (primary + N mirrors).

    Use this when a single event should land in BOTH the main log AND
    a per-subsystem dedicated log (camera.log today; motor.log /
    led.log / ids.log when those land). The first non-None logger
    argument is the primary; subsequent non-None ones are mirrors. If
    a mirror raises, the primary's call still landed and the failure
    is downgraded to a ``primary.debug()`` line so caller control flow
    stays intact.

    Args:
        *loggers: One or more logger-like objects exposing methods
            named by ``level`` (``logging.Logger`` instances qualify,
            as do test doubles). ``None`` entries are skipped so
            callers can pass an optional mirror without a guard. At
            least one non-None primary is required; with zero
            primaries the call is a silent no-op.
        level: ``'info'`` / ``'warning'`` / ``'error'`` / ``'debug'`` /
            ``'critical'`` / ``'exception'``. Same set as Python
            logging level-method names.
        message: The full log message. Convention: include the
            subsystem prefix (``'[CAM Class ]'``, ``'[XYZ Class ]'``,
            ``'[LED Class ]'``) so the line is grep-friendly across
            files.

    Canonical example -- the camera dual-write that
    ``pyloncamera._log_cam`` wraps over::

        log_to(logger, _cam_log, level='info',
               message='[CAM Class ] Disconnected from Pylon camera')
    """
    primary = None
    mirrors = []
    for lg in loggers:
        if lg is None:
            continue
        if primary is None:
            primary = lg
        else:
            mirrors.append(lg)
    if primary is None:
        return
    getattr(primary, level)(message)
    for mirror in mirrors:
        try:
            getattr(mirror, level)(message)
        except Exception:
            primary.debug(f'log_to: mirror.{level}() raised')
