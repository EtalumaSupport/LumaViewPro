# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A spinner that reports a person's pick without taking it as its value."""

from kivy.uix.spinner import Spinner


class PickSpinner(Spinner):
    """A spinner whose text is display only; a pick is an event.

    A plain Spinner writes the picked value into ``text``, and ``on_text``
    fires for that write and for every programmatic one alike, so a
    handler on ``on_text`` cannot tell a person's choice from the display
    being refreshed. Here a pick dispatches ``on_pick`` and leaves ``text``
    alone: whatever the pick changes, the display shows it only once the
    owner of that value has accepted it.
    """

    __events__ = ('on_pick',)

    def _on_dropdown_select(self, instance, data, *largs):
        self.is_open = False
        self.dispatch('on_pick', data)

    def on_pick(self, value):
        pass
