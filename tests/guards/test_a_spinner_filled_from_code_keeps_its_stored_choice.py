# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A spinner whose choices are filled from code does not pick the first of them.

A Kivy spinner with ``text_autoupdate`` set changes its text to its first
choice when its choices are set and its text is not among them. A panel that
fills a spinner's choices from code and then shows the stored choice fires the
spinner's handler in between with that first choice, and a handler that
writes what it is handed writes it over the stored one: every start reset the
live and sequenced image formats to TIFF and the z-stack's reference to Top.
The loader sets the text itself right after the choices, so such a spinner
never needs the first choice picked for it. Kivy is not importable in the test
environment, so the kv source is the contract.
"""

import re

from tests.ast_seams import REPO_ROOT

_FILLED = re.compile(r"ids(?:\['(\w+)'\]|\.(\w+))\.values\s*=")
_ID = re.compile(r'^(\s*)id:\s*(\w+)\s*$')
_AUTOUPDATE = re.compile(r'^(\s*)text_autoupdate:\s*True\b')


def _filled_from_code() -> set[str]:
    filled = set()
    for path in (REPO_ROOT / 'ui').rglob('*.py'):
        for match in _FILLED.finditer(path.read_text(encoding='utf-8')):
            filled.add(match.group(1) or match.group(2))
    return filled


def _autoupdating() -> dict[str, str]:
    """Each kv widget id that sets ``text_autoupdate: True``, to where it does."""
    found = {}
    for path in (REPO_ROOT / 'ui').rglob('*.kv'):
        widget_id, id_indent = None, None
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            named = _ID.match(line)
            if named:
                id_indent, widget_id = named.group(1), named.group(2)
                continue
            auto = _AUTOUPDATE.match(line)
            if auto and widget_id is not None and auto.group(1) == id_indent:
                found[widget_id] = f'{path.relative_to(REPO_ROOT)}:{number}'
    return found


def test_no_spinner_filled_from_code_picks_its_first_choice():
    filled = _filled_from_code()
    assert 'zstack_spinner' in filled, 'the scan of ui/ lost its way'
    offenders = sorted(
        f'{widget} ({where})' for widget, where in _autoupdating().items() if widget in filled
    )
    assert not offenders, (
        'These spinners are filled from code and set text_autoupdate: True, so filling '
        'them fires their handler with the first choice before the stored one is shown; '
        'drop text_autoupdate:\n  ' + '\n  '.join(offenders)
    )
