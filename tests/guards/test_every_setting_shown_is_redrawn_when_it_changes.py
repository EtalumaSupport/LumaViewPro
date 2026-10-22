# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting the window shows is redrawn when the store changes it, whoever wrote it.

The window follows the store through one table (``ui.listener_bridge``):
a changed path's prefix names the redraw that shows it. With REST served
beside the window, a setting no row routes would go on showing the value a
client replaced, so every template leaf is routed by a row, or named as one
the window does not show. A setting added without either fails the build.
"""

from modules import common_utils
from modules.settings_paths import _leaf_paths
from tests.settings_fixtures import shipped_template
from ui import listener_bridge


def test_every_template_leaf_is_routed_or_named_as_not_shown():
    routed = {prefix for prefix, _ in listener_bridge.SHOWN_SETTINGS}
    layers = set(common_utils.get_layers())

    def covered(path: str, table) -> bool:
        first = path.split('.')[0]
        # A layer's own settings are routed by the layer row.
        candidates = ('<layer>' + path[len(first) :],) if first in layers else ()
        return any(
            path == p or path.startswith(f'{p}.') or c == p or c.startswith(f'{p}.')
            for p in table
            for c in (*candidates, path)
        )

    unrouted = [
        path
        for path, _ in _leaf_paths(shipped_template())
        if not covered(path, routed) and not covered(path, listener_bridge.NOT_SHOWN)
    ]
    shown_and_not = sorted(routed & set(listener_bridge.NOT_SHOWN))

    assert unrouted == [], f'route these in SHOWN_SETTINGS or name them in NOT_SHOWN: {unrouted}'
    assert shown_and_not == [], shown_and_not
