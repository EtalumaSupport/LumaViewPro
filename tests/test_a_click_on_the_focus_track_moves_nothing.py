# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A click on the focus slider's track moves nothing; only a drag of its handle moves Z.

The focus slider spans the whole Z travel, about 14 mm, and Kivy's slider
jumps its value to wherever the track is touched, so one stray click beside
the arrow buttons drove the objective that far, toward the sample. A slider
that takes touches on its handle alone (Kivy's ``sensitivity: 'handle'``)
ignores a touch on the track: it is not grabbed, so it never releases, and
the release is what moves Z (``on_release: root.set_position``).
"""

import re

from tests.test_issue_166_kv_lock_bindings import _block_for_id


def test_the_focus_slider_takes_touches_on_its_handle_only():
    # pin-justified: kv is declarative source with no headless seam (the suite
    # mocks Kivy); the sim walk presses the track and reads Z unmoved.
    block = _block_for_id('obj_position')
    sensitivity = [line.strip() for line in block if re.match(r'\s*sensitivity:', line)]
    assert sensitivity == ["sensitivity: 'handle'"], sensitivity
    assert any('on_release: root.set_position' in line for line in block), (
        'the focus slider no longer moves Z on its release'
    )
