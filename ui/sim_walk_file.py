"""A sim walk file: the steps ``--sim-walk`` performs, read and checked whole.

Read in ``lumaviewpro.py`` beside ``--simulate``, before Kivy is imported, so
a malformed walk stops the launch before anything starts; hence no Kivy
import here. The driver that performs the steps is ``ui.sim_walk``.

A walk is a JSON list of steps, each an object with ``do`` naming the action
and the keys that action takes. Any step may carry ``settle_s`` (seconds to
let the app answer before the next step, default 0.3) and ``note`` (free text
carried into the record).
"""

import inspect
import json
import pathlib

from modules.finite_number import is_finite_number

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
    'hold': (frozenset({'seconds'}), frozenset()),
    'inject': (frozenset({'fault'}), frozenset({'args'})),
    'close': (frozenset(), frozenset()),
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
        if step['do'] == 'quit' and number != len(steps):
            raise WalkFileError(f'{source}: step {number}: quit ends a walk; it is the last step')
    return steps


def ends_in_quit(steps: list[dict]) -> bool:
    """Whether the walk stops the app at its end: one that ends in ``quit``."""
    return bool(steps) and steps[-1]['do'] == 'quit'


def unattended(steps: list[dict]) -> bool:
    """Whether the walk is run with nobody at the screen.

    One that ends in ``quit``, or that closes the window itself (``close``):
    either ends the app, so a walk that stops at a failed step stops the app
    too rather than leave it open behind a popup nobody will answer.
    """
    return ends_in_quit(steps) or any(step['do'] == 'close' for step in steps)


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
    for key in ('path', 'text', 'value', 'button', 'file', 'name', 'fault'):
        if key in step and not isinstance(step[key], str):
            raise WalkFileError(f'{where}: {key} is text')
    # JSON reads Infinity, and a bool is an int to Python: neither is a wait.
    for key in ('settle_s', 'timeout_s', 'seconds'):
        if key in step and not (is_finite_number(step[key]) and step[key] >= 0):
            raise WalkFileError(f'{where}: {key} is a finite number of seconds, not negative')
    if action == 'inject':
        _check_fault(step, where)


def injectable_faults() -> dict:
    """The simulator faults a walk may inject, by name: the writes REST serves.

    ``SimulationAPI``'s members marked for the wire, not in-process, and not
    reads, as their own marks say; the walk adds no rule of its own. Imported
    here rather than at the top: the scope's modules load the project's
    logger, which a walk that injects nothing need not load before the app
    sets up.
    """
    from modules import api_surface
    from modules.lumascope_api.simulation import SimulationAPI

    return {
        name: member
        for name, member in vars(SimulationAPI).items()
        if api_surface.mark_of(member) == api_surface.API and not api_surface.reads_of(member)
    }


def _check_fault(step: dict, where: str) -> None:
    # Checked here, so a misspelt fault or argument stops the launch, not the
    # walk at its step after whatever ran before it. The values are the
    # member's door's to judge when the step calls it.
    args = step.get('args', {})
    if not isinstance(args, dict):
        raise WalkFileError(f"{where}: 'inject' args is an object of the fault's arguments")
    faults = injectable_faults()
    fault = step['fault']
    if fault not in faults:
        raise WalkFileError(
            f'{where}: unknown fault {fault!r} (one of {", ".join(sorted(faults))})'
        )
    # The fault's parameters after self; none takes *args or **kwargs.
    parameters = list(inspect.signature(faults[fault]).parameters.values())[1:]
    missing = [p.name for p in parameters if p.default is p.empty and p.name not in args]
    if missing:
        raise WalkFileError(f'{where}: {fault!r} needs {", ".join(missing)}')
    unknown = sorted(args.keys() - {p.name for p in parameters})
    if unknown:
        raise WalkFileError(f'{where}: unknown argument {unknown[0]!r} for {fault!r}')
