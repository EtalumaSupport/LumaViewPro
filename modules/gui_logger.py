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


def button(name, detail=''):
    """Log a button press."""
    _log.info(f'BUTTON {name} {detail}')


def toggle(name, state):
    """Log a toggle state change."""
    _log.info(f'TOGGLE {name} {"ON" if state else "OFF"}')


def slider(name, value):
    """Log a slider value change."""
    _log.info(f'SLIDER {name} {value}')


def select(name, value):
    """Log a selection change (spinner, dropdown, etc.)."""
    _log.info(f'SELECT {name} {value}')


def frame_size(width, height, binning):
    """Log a framing change -- the displayed (post-binning) frame size + binning.

    Wired from ``MicroscopeSettings._apply_displayed_frame``, the single
    chokepoint both the frame-field edit (``frame_size``) and the binning
    toggle (``select_binning_size``) flow through, so one call covers every
    framing change the user makes -- including the frame-box resize that was
    previously absent from the GUI log.
    """
    _log.info(f'FRAME_SIZE {width}x{height} binning={binning}')


def protocol_action(action, detail=''):
    """Log a protocol-level action (run, stop, pause, step add, etc.)."""
    _log.info(f'PROTOCOL {action} {detail}')


def one_line(text: object) -> str:
    """Collapse free text to a single physical log line.

    Popup and notification messages are caller-supplied prose that may
    span paragraphs; written raw, the continuation lines carry no
    level/timestamp prefix and every line-oriented consumer of the log
    miscounts. Escaping (rather than indenting) keeps the record one
    physical line and round-trippable.
    """
    return str(text).replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')


def notification(severity: str, title: str, message: str, source: str = '') -> None:
    """Log every popup / notification the user sees.

    Wired from every path that produces visible UI:
    - ``modules.notification_center.NotificationCenter.notify`` -- every
      ``notifications.warning/error/critical`` call (which reaches the
      listener-registered popup bridge in lumaviewpro.py:on_start).
    - ``ui.notification_popup`` helpers for direct popup calls
      (``show_notification_popup``, ``show_confirmation_popup``,
      ``show_confirmation_w_ack_popup``).
    - Engineering plugin and other modal-prompt entry points, via their
      use of the canonical ``ui.notification_popup`` helpers.

    Pipe character separates fields so log-scrapers can split cleanly
    when titles or messages contain colons.
    """
    sev_str = severity if isinstance(severity, str) else str(severity)
    src_suffix = f' from={source}' if source else ''
    _log.info(f'NOTIFICATION {sev_str} | {one_line(title)} | {one_line(message)}{src_suffix}')


def popup_response(title: str, response: str) -> None:
    """Log the user's response to a modal popup (OK / Cancel / Ack / dismiss).

    Pairs with ``notification`` -- one entry when the popup is shown,
    one when the user resolves it. Without the response, post-mortem
    can tell what the user saw but not what they did with it.
    """
    _log.info(f'POPUP_RESPONSE {one_line(response)} | {one_line(title)}')


def text_input(name: str, value: object) -> None:
    """Log a text field's final committed value.

    Callers debounce before calling: ``on_text`` fires per keystroke, and
    only the value the user stopped on belongs in gui_interactions.log.
    The debounce needs a GUI timer, so it lives GUI-side; the log line's
    shape stays here with every other entry in this module.

    ``value`` is whatever the field holds -- text from the protocol fields,
    a parsed number from the video ones -- and is only interpolated, so it
    is typed by what this needs of it rather than by today's callers.
    """
    _log.info(f'TEXT_INPUT {name} {value}')


def window_event(event_name: str, detail: str = '') -> None:
    """Log a Kivy Window-level lifecycle event.

    Captures the events that the OS / window manager / global keyboard
    shortcuts deliver outside any registered widget -- the events that
    would otherwise leave a gap when reading the GUI log to reconstruct
    "what triggered shutdown / minimize / focus change?" Wired from the
    Window.bind sites in ``lumaviewpro.py``.

    Event names (kebab-cased for stable log-scraping):
    - ``close-requested`` -- ``on_request_close`` fired; the close
      sequence about to start. Detail includes ``protocol_running``.
    - ``close`` -- ``on_close`` fired; the window is closing for real.
    - ``minimize`` / ``maximize`` / ``restore`` -- window-state change.
    - ``focus`` -- focus gained or lost. Detail includes ``focused``.
    - ``keyboard`` -- a non-widget-consumed key event (Alt-F4 etc.).
      Detail names the key + modifiers.
    """
    detail = (detail or '').strip()
    suffix = f' {detail}' if detail else ''
    _log.info(f'WINDOW {event_name}{suffix}')
