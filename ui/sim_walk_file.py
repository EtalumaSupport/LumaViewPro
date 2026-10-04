"""A sim walk file: the steps ``--sim-walk`` performs, read and checked whole.

Read in ``lumaviewpro.py`` beside ``--simulate``, before Kivy is imported, so
a malformed walk stops the launch before anything starts; hence no Kivy
import here. The driver that performs the steps is ``ui.sim_walk``.

A walk is a JSON list of steps, each an object with ``do`` naming the action
and the keys that action takes. Any step may carry ``settle_s`` (seconds to
let the app answer before the next step, default 0.3) and ``note`` (free text
carried into the record).
"""

import json
import pathlib

DEFAULT_SETTLE_S = 0.3

# For each action: the keys it needs, and the ones it may carry.
_ACTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    'press': (frozenset({'path'}), frozenset()),
    'type': (frozenset({'path', 'text'}), frozenset({'commit'})),
    'select': (frozenset({'path', 'value'}), frozenset()),
    'answer': (frozenset({'button'}), frozenset({'optional', 'timeout_s'})),
    'choose': (frozenset({'path'}), frozenset({'file', 'cancel'})),
    'read': (frozenset({'path', 'props'}), frozenset()),
    'shot': (frozenset({'name'}), frozenset()),
    'wait': (frozenset(), frozenset({'timeout_s'})),
    'quit': (frozenset(), frozenset()),
}
_EVERY_STEP = frozenset({'do', 'settle_s', 'note'})


class WalkFileError(ValueError):
    """A walk file that cannot be performed as written; the message names the step."""


def take_walk_flag(argv: list[str], *, simulate: bool) -> tuple[pathlib.Path, list[dict]] | None:
    """The walk ``--sim-walk=<file>`` names, as (its resolved path, its steps).

    Every ``--sim-walk=`` argument is removed from ``argv`` first, refused or
    not, because Kivy exits on a flag it does not know. None when none is given.

    Raises:
        WalkFileError: given more than once, or without ``simulate`` (a walk on
            hardware would move a real stage).
        OSError, ValueError: from ``read_walk``.
    """
    flags = [arg for arg in argv if arg.startswith('--sim-walk=')]
    for flag in flags:
        argv.remove(flag)
    if not flags:
        return None
    if len(flags) > 1:
        raise WalkFileError('give --sim-walk once')
    if not simulate:
        raise WalkFileError('--sim-walk needs --simulate')
    path = pathlib.Path(flags[0].split('=', 1)[1]).resolve()
    return path, read_walk(path)


def read_walk(path: pathlib.Path) -> list[dict]:
    """The steps in the walk file at ``path``.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not JSON (``json.JSONDecodeError``), or a step
            is malformed (``WalkFileError``).
    """
    return parse_walk(pathlib.Path(path).read_text(), source=str(path))


def parse_walk(text: str, *, source: str) -> list[dict]:
    """The steps in a walk's JSON text, each checked against its action.

    Raises:
        json.JSONDecodeError: the text is not JSON.
        WalkFileError: not a list of steps, or a step whose action is unknown,
            lacks a key it needs, or carries one it does not take.
    """
    steps = json.loads(text)
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        raise WalkFileError(f'{source}: a walk is a list of steps, each a JSON object')
    for number, step in enumerate(steps, start=1):
        _check_step(step, f'{source}: step {number}')
    return steps


def _check_step(step: dict, where: str) -> None:
    action = step.get('do')
    if action not in _ACTIONS:
        raise WalkFileError(
            f'{where}: unknown action {action!r} (one of {", ".join(sorted(_ACTIONS))})'
        )
    needs, may = _ACTIONS[action]
    missing = sorted(needs - step.keys())
    if missing:
        raise WalkFileError(f'{where}: {action!r} needs {", ".join(missing)}')
    unknown = sorted(step.keys() - needs - may - _EVERY_STEP)
    if unknown:
        raise WalkFileError(f'{where}: unknown key {unknown[0]!r} for {action!r}')
    if action == 'choose' and ('file' in step) == bool(step.get('cancel')):
        raise WalkFileError(f"{where}: 'choose' needs file or cancel: true, not both")
    if action == 'read' and not (
        isinstance(step['props'], list)
        and step['props']
        and all(isinstance(p, str) for p in step['props'])
    ):
        raise WalkFileError(f"{where}: 'read' props is a non-empty list of property names")
    for key in ('path', 'text', 'value', 'button', 'file', 'name'):
        if key in step and not isinstance(step[key], str):
            raise WalkFileError(f'{where}: {key} is text')
    for key in ('settle_s', 'timeout_s'):
        if key in step and not (isinstance(step[key], (int, float)) and step[key] >= 0):
            raise WalkFileError(f'{where}: {key} is a number of seconds, not negative')
