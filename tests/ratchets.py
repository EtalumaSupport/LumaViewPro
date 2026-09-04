# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The suite's ratchets, announced at the end of every run.

A ratchet is a test that pins a count the codebase is migrating DOWN: a
number that may fall and may not rise (a ceiling), or that must equal
the tree so a fall lowers the pin in the same commit (an equality). Each
ratchet test module registers its measurement here at import, and
`conftest.pytest_terminal_summary` prints every registered ratchet's
current value beside its pin when the run ends, so the progress of each
migration is visible in every pytest log without opening a test file.

Only the ratchets whose module was collected appear: a full run lists
them all, a targeted run lists the ones it imported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Ratchet:
    name: str
    measure: Callable[[], int]
    pin: int
    # 'ceiling': the count may fall freely and may not rise.
    # 'equal': the count must equal the pin; a fall lowers the pin.
    rule: str


_REGISTRY: list[Ratchet] = []


def register(name: str, measure: Callable[[], int], pin: int, rule: str = 'ceiling') -> None:
    if rule not in ('ceiling', 'equal'):
        raise ValueError(f'unknown ratchet rule {rule!r}')
    _REGISTRY.append(Ratchet(name, measure, pin, rule))


def registered() -> list[Ratchet]:
    return list(_REGISTRY)


def summary_lines() -> list[str]:
    """One line per registered ratchet: name, current count, pin, rule."""
    lines = []
    width = max((len(r.name) for r in _REGISTRY), default=0)
    for ratchet in _REGISTRY:
        try:
            now = ratchet.measure()
        except Exception as exc:  # a broken measure must not hide the others
            lines.append(f'{ratchet.name:<{width}}  unmeasurable: {exc!r}')
            continue
        if now == ratchet.pin:
            state = 'at pin'
        elif now < ratchet.pin:
            state = f'{ratchet.pin - now} below pin' + (
                ' -- lower it' if ratchet.rule == 'equal' else ''
            )
        else:
            state = f'{now - ratchet.pin} OVER pin'
        lines.append(
            f'{ratchet.name:<{width}}  {now:>5}  (pin {ratchet.pin}, {ratchet.rule}; {state})'
        )
    return lines
