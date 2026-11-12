"""The sim walk driver: performs a walk's steps on the running app's widgets.

A sim walk is the list of GUI steps that runs a change in the real app,
against the simulator or a real scope. ``--sim-walk=<file>`` hands the
mechanical steps to this driver, so the person at the screen keeps only the
ones that need a person. It acts as a person does and decides nothing: every
press is a real touch at the control's place in the window, so the control's
own handlers, its ``gui_logger`` record and the focus commit of a field being
left all happen as for a mouse. Kivy's ``trigger_action`` would not do: a
drawer records only from its touch handler, and a trigger leaves a focused
field uncommitted.

A control is named by its place on screen, not by a kv rule: one rule can be
instantiated many times (a ``ChannelControl`` per channel), so the first segment
of a path names a live widget class that must have exactly one instance, and
each later segment walks ``.ids`` from there (``ImageSettings/BF/acquire_image``),
or, for a control with no id, ``Class[attr=value]`` among the descendants
(``ProtocolSettings/FileChooseBTN[context=load_protocol]``). A name that
matches nothing, or more than one live widget, stops the walk.

A step that cannot be performed, or whose touch did not reach its control,
stops the walk there with the reason; a later step never runs. ``close``
presses the window's X: the app's own close gate asks, and later steps answer
it. A walk that ends in ``quit`` stops the app at its end as a platform quit
does, through ``on_stop``, never through the gate, which would ask with nobody
there to answer. A walk that ends in ``quit`` or presses ``close`` is one
nobody watches, so a stop at a failed step also stops the app, with a picture
of the window; any other walk leaves the app open. Each step's outcome is logged to the main log under ``[SIM WALK  ]``.

``wait`` ends when nothing a gesture started is still in flight: no progress
popup is open and the app's ``in_flight`` names nothing (what holds the
scope, a lane with a task running or queued, an axis moving, an autofocus
sweep). Each owner answers for its own work; the walk asks and decides
nothing. A ``wait`` that times out names what it was still waiting on, so a
stuck walk says which. Any other popup is a question only a person answers,
so a ``wait`` that finds one stops the walk at once, as a press under it
does; a touch that closes a popup waits until it is gone, so the next step
never meets one the walk has answered. A turret pick, a jog, a go-to, a build: the step after the
``wait`` reads the state after the work, however long it took.

Two steps do what the world does rather than a person. ``hold`` sleeps the
UI thread for its ``seconds`` in one call, so the app meets a held thread
exactly as a stall makes one. ``inject`` calls one of the simulator's faults
(``scope.simulation``) by name with its ``args``, as a cable pulled at the
bench; a refusal from the fault stops the walk at that step. A walk with
an ``inject`` is refused at launch without ``--simulate``.

Walk files are read and checked by ``ui.sim_walk_file``.
"""

import logging
import pathlib
import time
import weakref

import numpy
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.graphics.opengl import (
    GL_PACK_ALIGNMENT,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    glPixelStorei,
    glReadPixels,
)
from kivy.tests.common import UnitTestTouch
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.dropdown import DropDown
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput

from _lumascope.modules import gui_logger
from _lumascope.modules.exceptions import Refusal
from ui import file_dialogs
from ui.progress_popup import CustomPopup
from ui.sim_walk_file import DEFAULT_SETTLE_S, ends_in_quit, unattended

logger = logging.getLogger('LVP.sim_walk')

# How long a touch may take to arrive: a ScrollView holds a touch back for its
# scroll_timeout before handing it to the child under it.
_TOUCH_ARRIVAL_S = 0.6
_ANSWER_TIMEOUT_S = 10.0
_OPTIONAL_ANSWER_TIMEOUT_S = 3.0
_WAIT_TIMEOUT_S = 30.0
_BRING_UP_TIMEOUT_S = 30.0
_POLL_S = 0.05
# The background a widget's picture keeps below the last row it draws.
_TRAILING_MARGIN_DP = 20
# What an action's generator gives when it has run out: the step is done.
_STEP_DONE = object()


class WalkStepError(Exception):
    """A step that cannot be performed as written; the message says why."""


class SimWalk:
    """Performs a checked list of walk steps on the Kivy thread, one at a time.

    ``bring_up_owes`` names what bring-up has not finished (the app passes
    its ``_bring_up_owes``): the walk starts once it names nothing, and stops
    before step 1, saying what was still owed, when it never does.
    ``in_flight`` names what a gesture started that is still running (the
    app passes the session's ``in_flight_named``): a ``wait`` ends once it names nothing and
    no popup is open. ``simulation`` is the scope's ``SimulationAPI``, which
    ``inject`` calls. A ``shot``'s name is a path under ``shot_dir`` (the
    walk file's folder); the picture of a stop goes to ``stop_dir`` (the
    run's log folder), beside the log line that names it. ``finished``,
    ``outcome`` (``'done'`` or ``'stopped at step N (...)'``), ``reads`` and
    ``shots`` are the result.
    """

    def __init__(
        self,
        steps: list[dict],
        *,
        source: str,
        bring_up_owes,
        in_flight,
        simulation,
        shot_dir: pathlib.Path,
        stop_dir: pathlib.Path,
    ):
        self._steps = steps
        self._source = source
        self._bring_up_owes = bring_up_owes
        self._in_flight = in_flight
        self._simulation = simulation
        self._bring_up_deadline = 0.0
        self._shot_dir = shot_dir
        self._stop_dir = stop_dir
        self._number = 0
        self._action = None
        self._ends_in_quit = ends_in_quit(steps)
        self._unattended = unattended(steps)
        # What the step's done line says, when it differs from 'done'.
        self._step_outcome = 'done'
        self.finished = False
        self.outcome = ''
        self.reads: list[dict] = []
        self.shots: list[str] = []

    def start(self) -> None:
        Window.bind(children=_watch_popups)
        _watch_popups()
        self._bring_up_deadline = time.monotonic() + _BRING_UP_TIMEOUT_S
        Clock.schedule_once(self._await_bring_up, 0)

    def _await_bring_up(self, _dt) -> None:
        owed = self._bring_up_owes()
        if owed:
            if time.monotonic() < self._bring_up_deadline:
                Clock.schedule_once(self._await_bring_up, 0.1)
                return
            self._finish(
                f'stopped before step 1: bring-up did not finish within '
                f'{_BRING_UP_TIMEOUT_S:g} s; still owed: {", ".join(owed)}'
            )
            return
        gui_logger.walk_scripted(self._source)
        logger.info(f'[SIM WALK  ] {self._source}: {len(self._steps)} steps')
        self._next(0)

    def _next(self, _dt) -> None:
        if self._number == len(self._steps):
            self._finish('done')
            return
        step = self._steps[self._number]
        self._number += 1
        self._step_outcome = 'done'
        self._action = getattr(self, f'_{step["do"]}')(step)
        self._resume(0)

    def _resume(self, _dt) -> None:
        step = self._steps[self._number - 1]
        try:
            delay = next(self._action, _STEP_DONE)
        # A step it cannot perform, or a refusal from what the step called
        # (an injected fault): the walk stops there, with its line.
        except (WalkStepError, Refusal) as e:
            self._finish(f'stopped at step {self._number} ({_describe(step)}): {e}')
            return
        if delay is _STEP_DONE:
            logger.info(f'[SIM WALK  ] step {self._number} {_describe(step)}: {self._step_outcome}')
            Clock.schedule_once(self._next, step.get('settle_s', DEFAULT_SETTLE_S))
            return
        Clock.schedule_once(self._resume, delay)

    def _finish(self, outcome: str) -> None:
        self.finished = True
        self.outcome = outcome
        if outcome == 'done':
            logger.info(f'[SIM WALK  ] {self._source}: done')
        else:
            logger.warning(f'[SIM WALK  ] {self._source}: {outcome}')
        if not self._unattended:
            return
        # Nobody is at the screen for this walk, so it never leaves the app
        # up when it stops: an open LVP blocks every other sim launch. The
        # picture keeps what a stop looked like. Done, a walk that pressed
        # ``close`` leaves the app to close itself.
        if outcome != 'done':
            if not self._picture(self._stop_dir / f'stopped_at_step_{self._number}.png'):
                logger.warning('[SIM WALK  ] the picture of the stop was not written')
            _stop_the_app()
        elif self._ends_in_quit:
            _stop_the_app()

    # --- the actions; each is a generator yielding the seconds to wait ------

    def _press(self, step):
        yield from self._touch(self._resolve(step['path']), step['path'])

    def _type(self, step):
        field = self._resolve(step['path'])
        if not isinstance(field, TextInput):
            raise WalkStepError(f'{step["path"]} is a {type(field).__name__}, not a text field')
        self._check_reachable(field, step['path'])
        field.focus = True
        yield 0
        field.select_all()
        field.delete_selection()
        # One character at a time through insert_text, as a person's keys arrive:
        # the field's input_filter judges each insertion whole, so the walk's
        # text in one call would be refused where a person's keystrokes are not.
        for character in step['text']:
            field.insert_text(character)
        # Left as a click away leaves it: a box commits only an edit, so text
        # equal to what the box held commits nothing, as for a person.
        if step.get('commit', True):
            field.focus = False

    def _select(self, step):
        spinner = self._resolve(step['path'])
        if not isinstance(spinner, Spinner):
            raise WalkStepError(f'{step["path"]} is a {type(spinner).__name__}, not a spinner')
        yield from self._touch(spinner, step['path'])
        deadline = time.monotonic() + _ANSWER_TIMEOUT_S
        while not spinner.is_open:
            if time.monotonic() > deadline:
                raise WalkStepError(f'{step["path"]}: its list did not open')
            yield _POLL_S
        options = [
            w
            for w in spinner._dropdown.container.children
            if getattr(w, 'text', None) == step['value']
        ]
        option = _one(options, repr(step['value']), f'{step["path"]} list')
        yield from self._touch(option, f'{step["path"]} option {step["value"]!r}')

    def _answer(self, step):
        optional = step.get('optional', False)
        timeout = step.get(
            'timeout_s', _OPTIONAL_ANSWER_TIMEOUT_S if optional else _ANSWER_TIMEOUT_S
        )
        deadline = time.monotonic() + timeout
        while True:
            # A person can touch only the popup on top; one under it is
            # answered by a later step, once the top one has closed.
            top = next(iter(open_popups()), None)
            buttons = [
                w
                for w in (top.walk(restrict=True) if top is not None else ())
                if isinstance(w, ButtonBehavior) and getattr(w, 'text', None) == step['button']
            ]
            if buttons:
                button = _one(buttons, repr(step['button']), 'the popup on top')
                yield from self._touch(button, f'popup button {step["button"]!r}')
                return
            if time.monotonic() > deadline:
                if top is None:
                    if optional:
                        self._step_outcome = 'no popup asked; optional, nothing pressed'
                        return
                    raise WalkStepError(
                        f'no open popup has a {step["button"]!r} button after {timeout:g} s'
                    )
                missing = (
                    f'the popup on top ({getattr(top, "title", "")!r}) has no '
                    f'{step["button"]!r} button'
                )
                if optional:
                    self._step_outcome = f'{missing}; optional, nothing pressed'
                    return
                raise WalkStepError(f'{missing} after {timeout:g} s')
            yield _POLL_S

    def _choose(self, step):
        button = self._resolve(step['path'])
        file_dialogs.answer_next_dialog('' if step.get('cancel') else step['file'])
        try:
            yield from self._touch(button, step['path'])
            deadline = time.monotonic() + _TOUCH_ARRIVAL_S
            while file_dialogs.scripted_answer_waiting():
                if time.monotonic() > deadline:
                    raise WalkStepError(f'{step["path"]}: the press opened no file dialog')
                yield _POLL_S
        finally:
            # Never leave an answer behind for a dialog a person opens later.
            file_dialogs.withdraw_scripted_answer()

    def _read(self, step):
        widget = self._resolve(step['path'])
        props = {}
        for prop in step['props']:
            if not hasattr(widget, prop):
                raise WalkStepError(f'{step["path"]} has no property {prop!r}')
            value = getattr(widget, prop)
            props[prop] = (
                value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            )
        # A control's properties keep their own dict, so a slider's `step`
        # never overwrites the walk's step number.
        record = {'step': self._number, 'path': step['path'], 'props': props}
        self.reads.append(record)
        logger.info(f'[SIM WALK  ] step {self._number} read {record}')
        yield from ()

    def _shot(self, step):
        # A popup fading out is a picture of neither its open nor its closed state.
        yield from _until_no_popup_fades()
        widget = None
        if 'path' in step:
            # A picture of a control the screen does not show is a picture of
            # something else, so it is refused where a press would be.
            widget = self._resolve(step['path'])
            self._check_reachable(widget, step['path'])
            yield from _until_drawers_settle(widget, step['path'])
        yield 0
        path = self._shot_dir / f'{step["name"]}.png'
        if not path.parent.is_dir():
            raise WalkStepError(f'the picture {step["name"]!r}: no folder {path.parent}')
        if widget is not None:
            for parent in _ancestors(widget):
                if isinstance(parent, ScrollView) and not _wholly_in_view(parent, widget):
                    raise WalkStepError(
                        f'{step["path"]} is not wholly in view in its scroll view; scroll first'
                    )
            if not _wholly_in_window(widget):
                raise WalkStepError(f'{step["path"]} is not wholly inside the window')
        if not self._picture(path, widget):
            raise WalkStepError(f'the window picture {step["name"]!r} was not written')

    def _picture(self, path: pathlib.Path, widget=None) -> bool:
        """Write what the window shows, or only ``widget``'s part of it, to exactly ``path``.

        Replaces a file already there; whether it was written. The whole
        window is drawn and read, so the picture is the screen, not the
        widget redrawn alone: a parent's background, a popup over the widget
        and what a scroll view clips are all as shown. A widget's picture ends
        a margin below the last row it draws: rows of its bottom background
        below that come off, so a drawer taller than its controls is pictured
        to its controls, whatever the window's height. Its width is never cut,
        so drawers of one width print at one scale.
        """
        width, height = Window.size
        # Between frames the back buffer holds whatever the last swap left,
        # which under load can be an empty frame; the window's own draw puts
        # the frame it is about to show there, and the read follows at once.
        Window.dispatch('on_draw')
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        # Rows from the bottom of the window up, as GL and a texture both hold them.
        pixels = numpy.frombuffer(
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE), dtype=numpy.uint8
        ).reshape(height, width, 3)
        if widget is not None:
            left, bottom, right, top = _window_bounds(widget)
            pixels = _without_trailing_background(pixels[bottom:top, left:right])
        rows, columns = pixels.shape[:2]
        texture = Texture.create(size=(columns, rows), colorfmt='rgb')
        texture.blit_buffer(pixels.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        if not CoreImage(texture).save(str(path), flipped=True):
            return False
        self.shots.append(str(path))
        logger.info(f'[SIM WALK  ] step {self._number} picture {path}')
        return True

    def _wait(self, step):
        timeout = step.get('timeout_s', _WAIT_TIMEOUT_S)
        deadline = time.monotonic() + timeout
        while True:
            # A progress popup closes itself when its work ends; any other is
            # a question only a person answers, so the walk stops at it now.
            questions = [view for view in open_popups() if not isinstance(view, CustomPopup)]
            if questions:
                raise WalkStepError(_answer_it_first(questions[0]))
            still = [f'a popup ({getattr(view, "title", "")!r})' for view in open_popups()]
            still.extend(
                f'a popup closing ({getattr(view, "title", "")!r})' for view in _fading_popups()
            )
            still.extend(self._in_flight())
            if not still:
                return
            if time.monotonic() > deadline:
                raise WalkStepError(f'still in flight after {timeout:g} s: {"; ".join(still)}')
            yield _POLL_S

    def _hold(self, step):
        started = time.monotonic()
        time.sleep(step['seconds'])
        self._step_outcome = f'held the UI thread {time.monotonic() - started:.1f} s'
        yield from ()

    def _inject(self, step):
        # The walk file has checked the fault and its arguments' names; a
        # refusal stops the walk in _resume.
        getattr(self._simulation, step['fault'])(**step.get('args', {}))
        yield from ()

    def _close(self, step):
        # The window's X, as a person presses it: the app's close gate logs
        # the request and asks or closes, and later steps answer it.
        Window.dispatch('on_request_close')
        yield from ()

    def _quit(self, step):
        # The walk's last step (the walk file holds it there); the stop is
        # the end of the walk's, in _finish.
        yield from ()

    # --- finding a control, and touching it ---------------------------------

    def _resolve(self, path: str):
        segments = path.split('/')
        widgets = [w for top in list(Window.children) for w in top.walk(restrict=True)]
        widget = _one(
            [w for w in widgets if _class_matches(w, segments[0])], repr(segments[0]), path
        )
        for segment in segments[1:]:
            if '[' in segment:
                found = [
                    w
                    for w in widget.walk(restrict=True)
                    if w is not widget and _class_matches(w, segment)
                ]
            else:
                found = [widget.ids[segment].__self__] if segment in widget.ids else []
            widget = _one(found, repr(segment), path)
        return widget

    def _check_reachable(self, target, path: str) -> None:
        if target.get_root_window() is None:
            raise WalkStepError(f'{path} is not on screen')
        for parent in _ancestors(target):
            if isinstance(parent, AccordionItem) and parent.collapse:
                raise WalkStepError(
                    f'{path} is inside the collapsed drawer {parent.title!r}; open it first'
                )
        popups = open_popups()
        if popups and not _contains(popups[0], _placed_by(target)):
            raise WalkStepError(_answer_it_first(popups[0]))

    def _touch(self, target, path: str):
        self._check_reachable(target, path)
        yield from _until_drawers_settle(target, path)
        # A popup fading out after its dismiss still takes every touch until
        # the fade removes it, so a press sent before then lands on it.
        yield from _until_no_popup_fades()
        # A person scrolls only to reach what they cannot see, so the walk
        # does too: Kivy's scroll_to moves content shorter than its view to
        # the view's bottom even when the target is already in it.
        scrolled = False
        for parent in _ancestors(target):
            if isinstance(parent, ScrollView) and not _wholly_in_view(parent, target):
                parent.scroll_to(target, padding=10, animate=False)
                scrolled = True
        if scrolled:
            yield 0
        x, y = target.to_window(*target.center)
        arrived = []

        def arrival(widget, touch):
            if widget.collide_point(*touch.pos):
                arrived.append(True)

        target.fbind('on_touch_down', arrival)
        try:
            touch = UnitTestTouch(x, y)
            touch.touch_down()
            touch.touch_up()
            deadline = time.monotonic() + _TOUCH_ARRIVAL_S
            while not arrived and time.monotonic() < deadline:
                yield _POLL_S
        finally:
            target.funbind('on_touch_down', arrival)
        if not arrived:
            raise WalkStepError(f'{path}: the touch at ({x:.0f}, {y:.0f}) did not reach it')


def _stop_the_app() -> None:
    """Stop the app as a platform quit does: its ``on_stop`` closes the session."""
    App.get_running_app().stop()


def _describe(step: dict) -> str:
    target = step.get('path') or step.get('button') or step.get('name') or step.get('fault') or ''
    return f'{step["do"]} {target}'.strip()


def _one(found: list, name: str, where: str):
    if not found:
        raise WalkStepError(f'{where}: no live widget matches {name}')
    if len(found) > 1:
        raise WalkStepError(f'{where}: {name} matches {len(found)} live widgets')
    return found[0]


def _class_matches(widget, segment: str) -> bool:
    """``Name`` is the widget's class; ``Name[attr=value]`` also needs the attribute."""
    name, _, condition = segment.partition('[')
    if type(widget).__name__ != name:
        return False
    if not condition:
        return True
    attr, _, value = condition.rstrip(']').partition('=')
    return str(getattr(widget, attr, None)) == value


def _answer_it_first(view) -> str:
    """Why a step stops at an open popup: the press's words and the wait's."""
    return f'a popup is open ({getattr(view, "title", "")!r}); answer it first'


# A popup told to close stays on the window while it fades out, until the
# fade removes it; a person sees it going, not open. Each is recorded as it
# is dismissed, and every popup is watched from the moment it reaches the
# window, so none closes unrecorded.
_closing: weakref.WeakSet = weakref.WeakSet()
_watched: weakref.WeakSet = weakref.WeakSet()


def _fading_popups() -> list:
    return [w for w in Window.children if w in _closing]


def _until_no_popup_fades():
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while fading := _fading_popups():
        if time.monotonic() > deadline:
            raise WalkStepError(
                f'the popup {getattr(fading[0], "title", "")!r} was dismissed but never closed'
            )
        yield _POLL_S


def _watch_popups(*_args) -> None:
    for view in Window.children:
        if isinstance(view, ModalView) and view not in _watched:
            _watched.add(view)
            view.fbind('on_pre_dismiss', _closing.add)
            view.fbind('on_pre_open', _closing.discard)


def open_popups() -> list:
    """The open popups, the one on top first: Kivy puts the newest window child at index 0.

    A popup fading out after its dismiss is not one: it asks nothing.
    """
    return [w for w in Window.children if isinstance(w, ModalView) and w not in _closing]


def _placed_by(widget):
    """The widget that put ``widget`` on screen.

    A dropdown's option is the dropdown's widget, but the list is added to the
    window rather than inside the spinner, so a spinner in a popup opens a list
    outside it; the option belongs to the popup through the spinner it opened
    from.
    """
    for parent in (widget, *_ancestors(widget)):
        if isinstance(parent, DropDown) and parent.attach_to is not None:
            return parent.attach_to
    return widget


def _contains(ancestor, widget) -> bool:
    return widget is ancestor or any(p is ancestor for p in _ancestors(widget))


def _moving_drawers(widget) -> list:
    """The drawers of every accordion around ``widget`` not yet drawn as their ``collapse`` says."""
    moving = []
    for parent in _ancestors(widget):
        if isinstance(parent, Accordion):
            moving.extend(
                item
                for item in parent.children
                if isinstance(item, AccordionItem) and item.collapse_alpha != float(item.collapse)
            )
    return moving


def _until_drawers_settle(target, path: str):
    # A drawer is where its state says it is only once its animation has
    # run and the accordion has laid out again; a touch sent before that
    # lands on whatever the stale layout put under it, and a picture shows
    # it half open. Bring-up opens the BF drawer in the frame the walk
    # starts, so the first press met this.
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while _moving_drawers(target):
        if time.monotonic() > deadline:
            raise WalkStepError(
                f'{path}: a drawer around it was still moving after {_WAIT_TIMEOUT_S:g} s'
            )
        yield _POLL_S
    # The accordion lays out on the frame after its last animation tick.
    yield _POLL_S


def _without_trailing_background(pixels: numpy.ndarray) -> numpy.ndarray:
    """``pixels`` (rows from the bottom up) less its bottom rows of one colour, all but a margin.

    The colour is the bottom-left pixel's; a bottom row with anything else in
    it trims nothing.
    """
    blank = (pixels == pixels[0, 0]).all(axis=(1, 2))
    # The first row with something drawn in it; 0 when the bottom row has, or every row is blank.
    drawn_from = int(numpy.argmin(blank))
    return pixels[max(0, drawn_from - round(dp(_TRAILING_MARGIN_DP))) :]


def _window_bounds(widget) -> tuple[int, int, int, int]:
    """The widget's (left, bottom, right, top) in window pixels, origin at the bottom left."""
    left, bottom = widget.to_window(*widget.pos)
    right, top = widget.to_window(widget.right, widget.top)
    return round(left), round(bottom), round(right), round(top)


def _wholly_in_window(widget) -> bool:
    left, bottom, right, top = _window_bounds(widget)
    width, height = Window.size
    return 0 <= left < right <= width and 0 <= bottom < top <= height


def _wholly_in_view(view: ScrollView, widget) -> bool:
    """Whether ``widget`` lies wholly inside the scroll view's visible area.

    Measured in the view's parent's coordinates, as Kivy's ``scroll_to``
    measures the same question.
    """
    left, bottom = view.parent.to_widget(*widget.to_window(*widget.pos))
    right, top = view.parent.to_widget(*widget.to_window(widget.right, widget.top))
    return view.x <= left and right <= view.right and view.y <= bottom and top <= view.top


def _ancestors(widget):
    """The widget's parents up to the window, which is its own parent in Kivy."""
    parent = widget.parent
    while parent is not None and parent is not Window:
        yield parent
        parent = parent.parent
