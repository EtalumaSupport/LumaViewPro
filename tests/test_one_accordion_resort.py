# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Both side panels put their accordion back in order the one way.

The left and right panels each carried a resort, the left one saying it
"mirrors" the right; a trap found in one (matching by Python id where
ids.get hands back a WeakProxy) was copied into the other by hand. They now
share resort_accordion: the shown items in canonical order, top first, then
any item the panel does not name -- a plugin's tab -- below them.
"""

from __future__ import annotations

import itertools

from ui.ui_helpers import resort_accordion

_uids = itertools.count()


class _Item:
    def __init__(self, name):
        self.name = name
        self.uid = next(_uids)
        self.parent = None


class _Accordion:
    """Kivy's children semantics: children[0] is drawn last, at the bottom;
    add_widget with no index prepends; remove_widget of a non-child is a
    no-op."""

    def __init__(self, *top_to_bottom):
        self.children = []
        for item in top_to_bottom:
            self.add_widget(item)

    def add_widget(self, widget, index=0):
        assert widget.parent is None, 'add_widget refuses an attached widget'
        self.children.insert(index, widget)
        widget.parent = self

    def remove_widget(self, widget):
        if widget not in self.children:
            return
        self.children.remove(widget)
        widget.parent = None

    def top_to_bottom(self):
        return [w.name for w in reversed(self.children)]


def test_shown_items_come_back_in_order_and_an_unnamed_one_goes_last():
    microscope, objective, xy, protocol, plugin = (
        _Item(n) for n in ('microscope', 'objective', 'xy', 'protocol', 'plugin')
    )
    accordion = _Accordion(protocol, plugin, microscope, xy, objective)

    resort_accordion(
        accordion,
        [(microscope, True), (objective, True), (xy, True), (protocol, True)],
    )

    assert accordion.top_to_bottom() == ['microscope', 'objective', 'xy', 'protocol', 'plugin']


def test_a_hidden_item_is_taken_out_and_a_missing_one_is_skipped():
    bf, pc, df = (_Item(n) for n in ('BF', 'PC', 'DF'))
    accordion = _Accordion(df, pc, bf)

    resort_accordion(accordion, [(bf, True), (pc, False), (None, True), (df, True)])

    assert accordion.top_to_bottom() == ['BF', 'DF']
    assert pc.parent is None


def test_a_shown_item_attached_elsewhere_is_moved_in():
    bf, green = _Item('BF'), _Item('Green')
    elsewhere = _Accordion(green)
    accordion = _Accordion(bf)

    resort_accordion(accordion, [(bf, True), (green, True)])

    assert accordion.top_to_bottom() == ['BF', 'Green']
    assert elsewhere.children == []


def test_both_panels_resort_through_the_helper(monkeypatch):
    from types import SimpleNamespace

    import ui.image_settings as image_settings
    import ui.motion_settings as motion_settings

    calls = []
    for module in (image_settings, motion_settings):
        monkeypatch.setattr(module, 'resort_accordion', lambda acc, items: calls.append(acc))
    monkeypatch.setattr(image_settings.common_utils, 'get_layers', lambda: ['BF'])

    left_accordion, right_accordion = object(), object()
    left = SimpleNamespace(
        ids={'motionsettings_accordion_id': left_accordion},
        _accordion_item_xystagecontrol=None,
        _accordion_item_xystagecontrol_visible=False,
        _LAYER_DISPLAY_ORDER=motion_settings.MotionSettings._LAYER_DISPLAY_ORDER,
    )
    right = SimpleNamespace(
        ids={'accordion_id': right_accordion},
        _resolve_pc_accordion=lambda: None,
        _accordion_item_df_control=None,
        _accordion_item_blue_control=None,
        _accordion_item_green_control=None,
        _accordion_item_red_control=None,
        _accordion_item_lumi_control=None,
        _accordion_item_pc_control_visible=False,
        _accordion_item_df_control_visible=False,
        _accordion_item_lumi_control_visible=False,
        _fluorescence_control_visible={},
    )

    motion_settings.MotionSettings._resort_accordion(left)
    image_settings.ImageSettings._resort_accordion(right)

    assert calls == [left_accordion, right_accordion]
