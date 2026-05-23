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


def test_status_bar_trigger_runs_at_at_least_10hz():
    """#638 follow-up: cursor XY readout cadence must be ≥ 10 Hz.

    Original implementation ran at 1 Hz which felt sluggish during
    stage motion — Eric flagged this on bench. The Clock.create_trigger
    interval must be ≤ 0.1 s so the title-bar XY position keeps up
    with motor moves.
    """
    import re

    src = _read('ui/shader.py')
    match = re.search(
        r'_status_bar_trigger\s*=\s*Clock\.create_trigger\('
        r'\s*self\._update_status_bar\s*,\s*([0-9]*\.?[0-9]+)',
        src,
    )
    assert match is not None, (
        '_status_bar_trigger Clock.create_trigger(...) call not found in ui/shader.py'
    )
    interval = float(match.group(1))
    assert interval <= 0.1, (
        f'_status_bar_trigger interval is {interval}s — must be '
        f'<= 0.1s (10 Hz) so cursor XY/Plate readouts stay responsive '
        f'during motion. (#638)'
    )
