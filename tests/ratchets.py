# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The suite's ratchets, announced at the end of every run.

A ratchet is a test that pins a count the codebase is migrating DOWN: a
number that may fall and may not rise (a ceiling), or that must equal
the tree so a fall lowers the pin in the same commit (an equality); an
announced count has no pin and is printed for its history. The ratchet
test observes the count it measured beside its pin, before its assert
(`observe`), and `conftest.pytest_terminal_summary` prints every
observation when the run ends, so the progress of each migration is
visible in every pytest log without opening a test file.

The ratchets whose test ran appear, failed or not: a full run lists them
all, a targeted run the ones it ran. Under xdist a worker's observations
travel to the controller in its `workeroutput` (`ship` at the worker's
session end, `take` as each node leaves, both from `tests/conftest.py`);
the controller imports no test module, which is why nothing measured at
import could reach it.
"""

from __future__ import annotations

from dataclasses import dataclass

RULES = ('ceiling', 'equal', 'announce')


@dataclass(frozen=True)
class Observation:
    name: str
    count: int
    pin: int | None  # None for an announced count
    # 'ceiling': the count may fall freely and may not rise.
    # 'equal': the count must equal the pin; a fall lowers the pin.
    # 'announce': no pin; the count is printed so its history is visible.
    rule: str


_OBSERVED: list[Observation] = []


def observe(name: str, count: int, pin: int | None = None, rule: str = 'ceiling') -> None:
    """Record the count a ratchet test measured, beside its pin.

    Called before the test's assert, so a ratchet that fails still prints.
    An announced count has no pin and a pinned count has one; either the
    other way round is refused here rather than printed wrong.
    """
    if rule not in RULES:
        raise ValueError(f'unknown ratchet rule {rule!r}')
    if (pin is None) != (rule == 'announce'):
        raise ValueError(f'{name}: rule {rule!r} with pin {pin!r}')
    _OBSERVED.append(Observation(name, count, pin, rule))


def ship() -> list[tuple]:
    """This process's observations as plain tuples, for a worker's workeroutput."""
    return [(o.name, o.count, o.pin, o.rule) for o in _OBSERVED]


def take(rows) -> None:
    """Observations a worker shipped, taken by the controller."""
    _OBSERVED.extend(Observation(*row) for row in rows)


def summary_lines() -> list[str]:
    """One line per observation, by name: name, count, pin, rule."""
    lines = []
    observed = sorted(_OBSERVED, key=lambda o: o.name)
    width = max((len(o.name) for o in observed), default=0)
    for o in observed:
        if o.rule == 'announce':
            lines.append(f'{o.name:<{width}}  {o.count:>5}  (announced, no pin)')
            continue
        if o.count == o.pin:
            state = 'at pin'
        elif o.count < o.pin:
            state = f'{o.pin - o.count} below pin' + (' -- lower it' if o.rule == 'equal' else '')
        else:
            state = f'{o.count - o.pin} OVER pin'
        lines.append(f'{o.name:<{width}}  {o.count:>5}  (pin {o.pin}, {o.rule}; {state})')
    return lines
