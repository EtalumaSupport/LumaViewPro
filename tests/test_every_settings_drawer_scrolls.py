# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every drawer of the left settings accordion scrolls its panel, and nothing in it stretches.

An open drawer has only the height the accordion leaves it: about 424dp at the
1024x600 minimum window, less with more drawers. A panel taller than that, with
no scroll view, squeezes its one stretching child to nothing and draws its
neighbours over each other: Objective Control's Z-stack over the turret
buttons. A child that stretches inside a scroll box is sized to nothing there,
so every child of the box declares its own height.

The post-processing drawer is an accordion of its own; its drawers are walked by
``test_every_post_processing_panel_is_scroll_wrapped``.

pin-justified: kv is declarative source with no headless seam; the kv text is
the contract.
"""

import re
from pathlib import Path

KV = (Path(__file__).resolve().parents[1] / 'ui' / 'lumaviewpro.kv').read_text(encoding='utf-8')
LINES = KV.split('\n')
_WIDGET = re.compile(r'^[A-Z]\w*:$')


def _indent(line: str) -> int:
    """Indentation width, as Kivy's parser reads it (a tab is 4)."""
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _children(at: int) -> list[tuple[str, dict[str, str], int]]:
    """The widgets directly under line *at*: each with its own properties and its line."""
    found = []
    base = _indent(LINES[at])
    for i in range(at + 1, len(LINES)):
        line = LINES[i]
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if _indent(line) <= base:
            break
        if _indent(line) == base + 4 and _WIDGET.match(text):
            found.append((text[:-1], {}, i))
        elif found and _indent(line) == base + 8 and ':' in text:
            key, value = text.split(':', 1)
            found[-1][1][key] = value.strip()
    return found


def _rule(name: str) -> int:
    return LINES.index(f'<{name}>:')


def _drawer_panels() -> list[str]:
    """The panel class in each drawer: the kv accordion's, and XY Stage's."""
    # The accordion's own lines carry its id at its properties' indent.
    accordion = next(i for i, line in enumerate(LINES) if 'id: motionsettings_accordion_id' in line)
    panels = []
    for i in range(accordion + 1, len(LINES)):
        text = LINES[i].strip()
        if text and not text.startswith('#') and _indent(LINES[i]) < _indent(LINES[accordion]):
            break
        if text == 'LoggedAccordionItem:':
            panels.append(_children(i)[0][0])
    panels.append(_children(_rule('AccordionItemXyStageControl'))[0][0])
    return panels


def _sizes_itself(widget: str, props: dict[str, str]) -> bool:
    if props.get('size_hint_y') == 'None':
        return True
    if f'<{widget}>:' in LINES:
        return dict(_children_props(_rule(widget))).get('size_hint_y') == 'None'
    return False


def _children_props(at: int) -> list[tuple[str, str]]:
    """A rule's own top-level properties."""
    base = _indent(LINES[at])
    props = []
    for line in LINES[at + 1 :]:
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if _indent(line) <= base:
            break
        if _indent(line) == base + 4 and ':' in text and not _WIDGET.match(text):
            key, value = text.split(':', 1)
            props.append((key, value.strip()))
    return props


def test_each_drawer_panel_scrolls_and_every_child_sizes_itself():
    panels = _drawer_panels()
    assert {'MicroscopeSettings', 'VerticalControl', 'XYStageControl'} <= set(panels), panels

    offenders = []
    for panel in panels:
        if panel == 'PostProcessingAccordion':
            continue
        scroll = [
            line
            for widget, _, line in _children(_rule(panel))
            if widget == 'ModSliderAwareScrollView'
        ]
        if not scroll:
            offenders.append(f'{panel}: no ModSliderAwareScrollView')
            continue
        [(_, box_props, box_line)] = _children(scroll[0])
        if box_props.get('height') != 'self.minimum_height':
            offenders.append(f'{panel}: the scroll box is not as tall as its children')
        offenders += [
            f'{panel}: {child} stretches inside the scroll box'
            for child, props, _ in _children(box_line)
            if not _sizes_itself(child, props)
        ]
    assert not offenders, '\n'.join(offenders)
