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

_write_backs: dict = {}


def note_write_back(name: str, value: object) -> None:
    """Declare that the APP is about to write, or has just written, ``value``.

    A widget cannot tell the difference between a user operating it and the app
    assigning to it -- both dispatch the same event, so the handler runs and a
    record comes out either way. Two shapes of that produce records nobody did:

    - a spinner whose options or text are set during panel setup, which
      dispatches once per assignment;
    - a text box whose handler corrects a typed value and writes the correction
      back into the box, so a record carrying the corrected value is the app's
      write and not the user's entry.

    The writer declares the value; the next record for that name carrying
    exactly it is recognised as the app's own and dropped. Exactly one is
    absorbed, so a user who then does the same thing deliberately still records.

    Declare BEFORE a write that dispatches synchronously (the spinner case) and
    AFTER one whose echo arrives later (the text case).
    """
    _write_backs[name] = str(value)


def consume_write_back(name: str, value: object) -> bool:
    """True when this record is the app's own write rather than a user action.

    Consumes the declaration either way, so a declaration that never matched
    cannot linger and swallow a later genuine record.
    """
    expected = _write_backs.pop(name, None)
    return expected is not None and str(value) == expected


def button(name, detail=''):
    """Log a button press."""
    _log.info(f'BUTTON {name} {detail}')


def toggle(name: str, state: bool) -> None:
    """Log a toggle state change.

    ``state`` must be a real ``bool``. The two toggle-ish Kivy widgets do not
    agree on how they expose their value: a ``CheckBox`` has ``active``, which
    is already a bool, while a ``ToggleButton`` has ``state``, which is the
    string ``'normal'`` or ``'down'`` -- and BOTH of those strings are truthy.
    A caller handing ``widget.state`` straight through would therefore log
    ``ON`` for every gesture including the ones turning the control off, and a
    record that is wrong is worse than one that is missing: nothing downstream
    can tell it from a real press. Callers convert at the call site with
    ``widget.state == 'down'``; this refuses the unconverted value rather than
    relying on each new caller to remember.
    """
    if not isinstance(state, bool):
        raise TypeError(
            f'gui_logger.toggle({name!r}, ...) needs a bool, got '
            f'{type(state).__name__} {state!r}. A ToggleButton exposes '
            f"'normal'/'down', both truthy -- convert with state == 'down'."
        )
    _log.info(f'TOGGLE {name} {"ON" if state else "OFF"}')


def slider(name, value):
    """Log a slider value change."""
    _log.info(f'SLIDER {name} {value}')


def select(name: str, value: object) -> None:
    """Log a selection change (spinner, dropdown, etc.).

    Spinners dispatch their text-change event on a programmatic assignment as
    well as on a user pick, so a panel that populates its own options would
    otherwise record a selection at every app start.
    """
    if consume_write_back(name, value):
        return
    _log.info(f'SELECT {name} {value}')


def frame_size(width: int, height: int, binning: int) -> None:
    """Log a framing change -- the displayed (post-binning) frame size + binning.

    Wired from ``MicroscopeSettings._framing_applied``, the redraw both the
    frame-field edit (``frame_size``) and the binning pick
    (``select_binning_size``) end in, so one call covers every framing change
    the user makes -- including the frame-box resize that was once absent
    from the GUI log. It records the framing stored once the camera answered.
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
    """Log a notification posted to the user, at its severity.

    Written by ``modules.notification_center.NotificationCenter.notify``,
    once per notification, whether or not it is shown; the popup that
    shows one is recorded separately by ``dialog``. The severity is the
    notification's, so this is the one GUI record that carries a level.

    Pipe character separates fields so log-scrapers can split cleanly
    when titles or messages contain colons.
    """
    sev_str = severity if isinstance(severity, str) else str(severity)
    src_suffix = f' from={source}' if source else ''
    _log.info(f'NOTIFICATION {sev_str} | {one_line(title)} | {one_line(message)}{src_suffix}')


def dialog(title: str, body: str) -> None:
    """Log a dialog as it opens: what was on the screen.

    Written from the one place every dialog opens (``ui.notification_popup``,
    the patched ``Popup.open``), whoever built it. A dialog has no severity:
    a question asks, and a notice's level is its notification's, recorded
    by ``notification``.
    """
    _log.info(f'DIALOG | {one_line(title)} | {one_line(body)}')


def popup_response(title: str, response: str) -> None:
    """Log the user's response to a modal popup (OK / Cancel / Ack / dismiss).

    Pairs with ``dialog`` -- one entry when the popup is shown,
    one when the user resolves it. Without the response, post-mortem
    can tell what the user saw but not what they did with it.
    """
    _log.info(f'POPUP_RESPONSE {one_line(response)} | {one_line(title)}')


def text_input(name: str, value: object) -> None:
    """Log a text field's final committed value.

    Call this once per commit, from the handler the box's ``on_focus`` binding
    reaches, with what the box holds BEFORE the handler transforms it. The
    emit is synchronous and unconditional: the record lands in the file ahead
    of whatever the entry goes on to do, so a bundle reads in the order the
    user acted, and the last thing typed before a freeze or a crash is already
    written.

    A handler that CORRECTS the entry -- a clamp, a sanitiser, a substitution
    for something unparseable -- records the correction under
    ``<name>_APPLIED``. The pair is the contract, and both halves are needed:
    alone, the first says the user asked for something the app never did, and
    the second asserts they typed a value they did not. Emit ``_APPLIED`` only
    when the value actually moved, comparing the PARSED values rather than the
    strings, or every float box reports '5' -> 5.0 as a correction.

    ``value`` is whatever the field holds -- text from the protocol fields,
    a parsed number from the video ones -- and is only interpolated, so it
    is typed by what this needs of it rather than by today's callers.
    """
    _log.info(f'TEXT_INPUT {name} {value}')


def walk_scripted(source: str) -> None:
    """Log that the presses which follow are a scripted sim walk's, not a person's.

    The walk driver (``ui.sim_walk``) touches widgets through the same handlers a
    person's touch reaches, so its presses record exactly as a person's would --
    that is what a walk checks. This one line, written before the first step, is
    what tells a reader of the log that no person made them.
    """
    _log.info(f'WALK SCRIPTED {source}')


_shown: dict[str, str] = {}


def display(name: str, value: object) -> None:
    """Log a change in what the window shows, as it changes.

    A press is recorded where the person made it; what the window then shows
    -- the controls greyed, the homing banner, the title's event text -- is
    decided by the app and was in no record, so whether it appeared could only
    be asked of whoever was watching. Recorded only when the value differs from
    the last one recorded for ``name``, so a writer that runs on every edge
    adds a line only when the screen changed.

    A count inside the value (a recording's elapsed seconds, a drain's files
    left) is progress, not a change of what is shown: it is compared with its
    digits blanked, so the record names each stage once.
    """
    shown = one_line(value)
    stage = ''.join('#' if c.isdigit() else c for c in shown)
    if _shown.get(name) == stage:
        return
    _shown[name] = stage
    _log.info(f'DISPLAY {name} {shown}')


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
