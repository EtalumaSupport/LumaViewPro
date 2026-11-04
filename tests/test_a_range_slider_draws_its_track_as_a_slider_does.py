# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A range slider draws its track as Kivy's own slider draws it, at every density.

The Object Analysis panel puts RangeSliders beside ModSliders. RangeSlider is
a hand fork of Kivy's slider; once its track box was halved with no border, its
track image drew inside out on a standard-density screen: a 1 px bar, or none,
beside the slider's 6 px. The child process renders both widgets under real
Kivy at density 1, the reporter's screen, and compares the track instructions
they built.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from kivy.graphics import BorderImage
from kivy.uix.slider import Slider
from ui.range_slider import RangeSlider

def tracks(widget):
    return [(tuple(i.border), tuple(i.size)) for i in widget.canvas.children
            if isinstance(i, BorderImage)]

print('TRACKS', [tracks(Slider(size=(300, 40))), tracks(RangeSlider(size=(300, 40)))])
"""


@pytest.mark.needs_display
def test_each_range_slider_track_has_the_slider_border_and_box_height(tmp_path):
    # Kivy reads its config from and writes its logs into KIVY_HOME, so the
    # child gets one of its own rather than the shared ~/.kivy.
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT, str(REPO)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'KIVY_HOME': str(tmp_path),
            'KIVY_METRICS_DENSITY': '1',
            'KIVY_NO_ARGS': '1',
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    [line] = [line for line in result.stdout.splitlines() if line.startswith('TRACKS ')]
    [slider_track], ranges = ast.literal_eval(line.removeprefix('TRACKS '))

    # The track and the connector between the handles; a border is
    # (bottom, right, top, left), and one wider than its box draws inside out.
    assert len(ranges) == 2, ranges
    for border, (width, height) in ranges:
        assert (border, height) == (slider_track[0], slider_track[1][1])
        assert border[0] + border[2] <= height and border[1] + border[3] <= width
