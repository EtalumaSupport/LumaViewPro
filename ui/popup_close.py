# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""App-wide close (X) button for Kivy popups.

Kivy's stock Popup ships no close affordance -- it relies entirely on
auto_dismiss (click-outside) plus whatever buttons the caller adds. This
module adds a title-row X to every Popup so they read like a normal
windowed dialog, without changing any popup's click-outside policy.
"""

from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

from lvp_logger import logger


def add_popup_close(popup) -> None:
    """Inject a top-right close (X) button into a Popup's title row.

    The X always dismisses, including popups that set auto_dismiss=False
    (the progress popup and error notifications, whose click-outside is
    deliberately disabled so a stray click cannot kill a long operation or
    drop an unread error). The X is their explicit escape hatch; it does
    not change the click-outside policy.

    Wired app-wide from a global '<Popup>: on_open' rule (below) instead of
    a kv child widget: Kivy merges the built-in <Popup> rule with ours, and
    a second top-level child is routed to `content`, raising PopupException
    ("Popup can have only one widget as content"). Injecting from on_open
    into the already-built internal GridLayout sidesteps that. Idempotent --
    on_open fires on every open.
    """
    if getattr(popup, '_etaluma_close_added', False):
        return
    container = getattr(popup, '_container', None)
    if container is None or container.parent is None:
        # Popup body not built yet -- no internal container to attach to.
        return

    # Internal structure built by Kivy's <Popup> rule: a GridLayout holding
    # the title Label, a separator Widget, and the content container.
    grid = container.parent
    title_label = next((c for c in grid.children if isinstance(c, Label)), None)
    if title_label is None:
        return

    popup._etaluma_close_added = True

    # Replace the bare title Label with a row that holds the title plus the
    # X, so the close button sits in the top-right corner on the title line.
    title_index = grid.children.index(title_label)
    grid.remove_widget(title_label)

    row = FloatLayout(size_hint_y=None, height=title_label.height)
    title_label.bind(height=lambda _inst, h: setattr(row, 'height', h))
    title_label.size_hint_x = 1
    title_label.pos_hint = {'x': 0, 'center_y': 0.5}
    row.add_widget(title_label)

    close_btn = Button(
        text='X',
        bold=True,
        size_hint=(None, None),
        size=(dp(26), dp(26)),
        pos_hint={'right': 1, 'center_y': 0.5},
        background_normal='',
        background_color=(0, 0, 0, 0),
    )
    close_btn.bind(on_release=lambda *_: popup.dismiss())
    row.add_widget(close_btn)

    grid.add_widget(row, index=title_index)
    logger.debug('[Popup    ] close button added to popup title row')


# Event-only rule: no kv child widget (see add_popup_close docstring for why
# a child would raise PopupException). Applies to every Popup app-wide,
# including subclasses (CustomPopup) and Popups built in pure Python.
Builder.load_string(
    """
#:import popup_close ui.popup_close
<Popup>:
    on_open: popup_close.add_popup_close(self)
"""
)
