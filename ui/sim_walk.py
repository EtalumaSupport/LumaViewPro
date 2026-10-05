"""The sim walk driver: performs a walk's steps on the running app's widgets.

A sim walk is the list of GUI steps that proves a change in the real app
against the simulator. ``--sim-walk=<file>`` (with ``--simulate``) hands the
mechanical steps to this driver, so the person at the screen keeps only the
ones that need a person. It acts as a person does and decides nothing: every
press is a real touch at the control's place in the window, so the control's
own handlers, its ``gui_logger`` record and the focus commit of a field being
left all happen as for a mouse. Kivy's ``trigger_action`` would not do: a
drawer records only from its touch handler, and a trigger leaves a focused
field uncommitted.

A control is named by its place on screen, not by a kv rule: one rule can be
instantiated many times (a ``LayerControl`` per layer), so the first segment
of a path names a live widget class that must have exactly one instance, and
each later segment walks ``.ids`` from there (``ImageSettings/BF/acquire_image``),
or, for a control with no id, ``Class[attr=value]`` among the descendants
(``ProtocolSettings/FileChooseBTN[context=load_protocol]``). A name that
matches nothing, or more than one live widget, stops the walk.

A step that cannot be performed, or whose touch did not reach its control,
stops the walk there with the reason; a later step never runs. A walk that
ends in ``quit`` is one nobody watches, so it closes the app at its end, done
or stopped, with a picture of the window at a stop; any other walk leaves the
app open. Each step's outcome is logged to the main log under ``[SIM WALK  ]``.

Walk files are read and checked by ``ui.sim_walk_file``.
"""

import logging
import pathlib
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.tests.common import UnitTestTouch
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.dropdown import DropDown
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput

from modules import gui_logger
from ui import file_dialogs
from ui.sim_walk_file import DEFAULT_SETTLE_S, closes_at_end

logger = logging.getLogger('LVP.sim_walk')

# How long a touch may take to arrive: a ScrollView holds a touch back for its
# scroll_timeout before handing it to the child under it.
_TOUCH_ARRIVAL_S = 0.6
_ANSWER_TIMEOUT_S = 10.0
_OPTIONAL_ANSWER_TIMEOUT_S = 3.0
_WAIT_TIMEOUT_S = 30.0
_BRING_UP_TIMEOUT_S = 30.0
_POLL_S = 0.05
# What an action's generator gives when it has run out: the step is done.
_STEP_DONE = object()


class WalkStepError(Exception):
    """A step that cannot be performed as written; the message says why."""


class SimWalk:
    """Performs a checked list of walk steps on the Kivy thread, one at a time.

    ``bring_up_owes`` names what bring-up has not finished (the app passes
    its ``_bring_up_owes``): the walk starts once it names nothing, and stops
    before step 1, saying what was still owed, when it never does.
    ``shot_dir`` receives the window pictures. ``finished``, ``outcome``
    (``'done'`` or ``'stopped at step N (...)'``), ``reads`` and ``shots`` are
    the result.
    """

    def __init__(self, steps: list[dict], *, source: str, bring_up_owes, shot_dir: pathlib.Path):
        self._steps = steps
        self._source = source
        self._bring_up_owes = bring_up_owes
        self._bring_up_deadline = 0.0
        self._shot_dir = shot_dir
        self._number = 0
        self._action = None
        self._closes_at_end = closes_at_end(steps)
        # What the step's done line says, when it differs from 'done'.
        self._step_outcome = 'done'
        self.finished = False
        self.outcome = ''
        self.reads: list[dict] = []
        self.shots: list[str] = []

    def start(self) -> None:
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
        except WalkStepError as e:
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
        if not self._closes_at_end:
            return
        # Nobody is at the screen for this walk, so it never leaves the app
        # up, done or stopped: an open LVP blocks every other sim launch. The
        # picture keeps what a stop looked like.
        if outcome != 'done' and not self._picture(f'stopped_at_step_{self._number}_'):
            logger.warning('[SIM WALK  ] the picture of the stop was not written')
        _close_the_app()

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
            buttons = [
                w
                for view in _open_popups()
                for w in view.walk(restrict=True)
                if isinstance(w, ButtonBehavior) and getattr(w, 'text', None) == step['button']
            ]
            if buttons:
                button = _one(buttons, repr(step['button']), 'the open popups')
                yield from self._touch(button, f'popup button {step["button"]!r}')
                return
            if time.monotonic() > deadline:
                if optional:
                    self._step_outcome = 'no popup asked; optional, nothing pressed'
                    return
                raise WalkStepError(
                    f'no open popup has a {step["button"]!r} button after {timeout:g} s'
                )
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
        record = {'step': self._number, 'path': step['path']}
        for prop in step['props']:
            if not hasattr(widget, prop):
                raise WalkStepError(f'{step["path"]} has no property {prop!r}')
            value = getattr(widget, prop)
            record[prop] = (
                value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            )
        self.reads.append(record)
        logger.info(f'[SIM WALK  ] step {self._number} read {record}')
        yield from ()

    def _shot(self, step):
        yield 0
        if not self._picture(step['name']):
            raise WalkStepError(f'the window picture {step["name"]!r} was not written')

    def _picture(self, name: str) -> bool:
        """Write the window to ``<shot_dir>/<name>NNNN.png``; whether it was written."""
        path = Window.screenshot(name=str(self._shot_dir / f'{name}.png'))
        if not path or not pathlib.Path(path).is_file():
            return False
        self.shots.append(str(path))
        logger.info(f'[SIM WALK  ] step {self._number} picture {path}')
        return True

    def _wait(self, step):
        timeout = step.get('timeout_s', _WAIT_TIMEOUT_S)
        deadline = time.monotonic() + timeout
        while _open_popups() or getattr(App.get_running_app(), 'run_lockout', False):
            if time.monotonic() > deadline:
                raise WalkStepError(f'a popup or a run was still up after {timeout:g} s')
            yield _POLL_S

    def _quit(self, step):
        # The walk's last step (the walk file holds it there); the close is
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
        popups = _open_popups()
        if popups and not _contains(popups[0], _placed_by(target)):
            raise WalkStepError(
                f'a popup is open ({getattr(popups[0], "title", "")!r}); answer it first'
            )

    def _touch(self, target, path: str):
        self._check_reachable(target, path)
        # A drawer is where its state says it is only once its animation has
        # run and the accordion has laid out again; a touch sent before that
        # lands on whatever the stale layout put under it. Bring-up opens the
        # BF drawer in the frame the walk starts, so the first press met this.
        deadline = time.monotonic() + _WAIT_TIMEOUT_S
        while _moving_drawers(target):
            if time.monotonic() > deadline:
                raise WalkStepError(
                    f'{path}: a drawer around it was still moving after {_WAIT_TIMEOUT_S:g} s'
                )
            yield _POLL_S
        # The accordion lays out on the frame after its last animation tick.
        yield _POLL_S
        scrolled = False
        for parent in _ancestors(target):
            if isinstance(parent, ScrollView):
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


def _close_the_app() -> None:
    """Close as the window's X does, so the app's own close gate still asks."""
    if not Window.dispatch('on_request_close'):
        App.get_running_app().stop()


def _describe(step: dict) -> str:
    target = step.get('path') or step.get('button') or step.get('name') or ''
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


def _open_popups() -> list:
    """The open popups, the one on top first: Kivy puts the newest window child at index 0."""
    return [w for w in Window.children if isinstance(w, ModalView)]


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


def _ancestors(widget):
    """The widget's parents up to the window, which is its own parent in Kivy."""
    parent = widget.parent
    while parent is not None and parent is not Window:
        yield parent
        parent = parent.parent
