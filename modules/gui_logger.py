"""GUI interaction logger for crash forensics and test creation.

Logs every user interaction BEFORE the action executes, so crash/freeze
forensics show exactly what the user did last. Also provides data for
creating automated tests from real user workflows.

WORKAROUND: INFO level during beta/early 4.0.x releases for maximum
visibility. Move to DEBUG level once crash/freeze issues are resolved.

Log file: logs/LVP_Log/gui_interactions.log (separate from main log)
"""

import logging

_log = logging.getLogger('LVP.gui_interactions')


def button(name, detail=""):
    """Log a button press."""
    _log.info(f"BUTTON {name} {detail}")


def toggle(name, state):
    """Log a toggle state change."""
    _log.info(f"TOGGLE {name} {'ON' if state else 'OFF'}")


def slider(name, value):
    """Log a slider value change."""
    _log.info(f"SLIDER {name} {value}")


def select(name, value):
    """Log a selection change (spinner, dropdown, etc.)."""
    _log.info(f"SELECT {name} {value}")


def protocol_action(action, detail=""):
    """Log a protocol-level action (run, stop, pause, step add, etc.)."""
    _log.info(f"PROTOCOL {action} {detail}")


def notification(severity, title, message, source=""):
    """Log every popup / notification the user sees.

    Crash-forensics gap (Eric 2026-05-04): the camera characterization
    run prompted "block the lens" twice; gui_interactions.log captured
    neither. Without the popup text in the log, post-mortem can't tell
    what the user was looking at when something went wrong.

    Wired from BOTH paths that produce visible UI:
    - ``modules.notification_center.NotificationCenter.notify`` — every
      ``notifications.warning/error/critical`` call (which reaches the
      listener-registered popup bridge in lumaviewpro.py:on_start).
    - ``ui.notification_popup`` helpers for direct popup calls
      (``show_notification_popup``, ``show_confirmation_popup``,
      ``show_confirmation_w_ack_popup``).

    Pipe character separates fields so log-scrapers can split cleanly
    when titles or messages contain colons.
    """
    sev_str = severity if isinstance(severity, str) else str(severity)
    src_suffix = f" from={source}" if source else ""
    _log.info(f"NOTIFICATION {sev_str} | {title} | {message}{src_suffix}")
