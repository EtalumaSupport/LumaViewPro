# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#638 regression: status bar must include cursor XY readouts when the
mouse hovers the live view.

Static-source check that ``_update_status_bar`` references
``_mouse_over_image`` and emits both ``Pixel:`` and ``Plate:`` strings.
The bug was that the cursor-XY block was missing entirely after the
d423d3c single-owner refactor, so the test must fail when the block
is missing.
"""
from pathlib import Path


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text()


def test_update_status_bar_emits_pixel_and_plate():
    src = _read('ui/shader.py')

    start = src.find('def _update_status_bar')
    assert start != -1, '_update_status_bar() not found in ui/shader.py'
    end = src.find('\n    def ', start + 1)
    body = src[start:end]

    assert '_mouse_over_image' in body, (
        '_update_status_bar must read _mouse_over_image to gate the '
        'cursor XY readouts. Without this gate, stale "0,0" coordinates '
        'show when the mouse is not over the image.'
    )
    assert 'Pixel:' in body, (
        '_update_status_bar must emit a "Pixel: (x, y)" component when '
        'the mouse is over the live view. (#638)'
    )
    assert 'Plate:' in body, (
        '_update_status_bar must emit a "Plate: (x, y) mm" component '
        'when the mouse is over the live view and motor is connected. '
        '(#638)'
    )
