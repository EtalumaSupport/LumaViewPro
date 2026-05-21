# Copyright Etaluma, Inc.
"""Regression tests for the 2026-05-14 least-astonishment cluster.

Three sites where user-visible text contradicted what the user could see on
screen:

1. Futures metrics log line used A/P/L letters that read as
   Active/Pending/Leaked; the actual fields are alloc/pop/live (monotonic
   counters). Rule 20: log line is ambiguous from two interpretations.
2. Post-processing empty-result popup said "No images found" in folders
   that visibly contained images; the structural cause was "no usable image
   groups for this operation" and the message is now PostFunction-specific.
3. Folder picker (tkinter askdirectory / osascript "choose folder") showed
   "no items match your search" in folders containing only image files;
   replaced with a Kivy popup that shows files too.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFuturesMetricsFormat:
    """The metric line names its fields rather than abbreviating to A/P/L."""

    def test_format_string_uses_explicit_field_names(self):
        src = (REPO_ROOT / "modules" / "config_helpers.py").read_text()
        m = re.search(
            r'futures_parts\.append\(\s*f"(?P<fmt>[^"]+)"\s*\)',
            src,
        )
        assert m is not None, "futures_parts append line not found"
        fmt = m.group("fmt")
        assert "alloc=" in fmt, f"alloc= missing from: {fmt!r}"
        assert "pop=" in fmt, f"pop= missing from: {fmt!r}"
        assert "live=" in fmt, f"live= missing from: {fmt!r}"

    def test_old_abbreviated_format_not_present(self):
        src = (REPO_ROOT / "modules" / "config_helpers.py").read_text()
        # The old format invited Active/Pending/Leaked misread; must not
        # come back in any form.
        assert "=A{" not in src
        assert "/P{" not in src
        assert "/L{" not in src


class TestPostProcessingEmptyResultMessages:
    """Empty-result popups name the missing capture dimension per PostFunction."""

    def test_multi_frame_requirement_covers_each_post_function(self):
        from modules.common_utils import PostFunction
        from modules.protocol_post_processor import _MULTI_FRAME_REQUIREMENT

        assert PostFunction.VIDEO in _MULTI_FRAME_REQUIREMENT
        assert "time point" in _MULTI_FRAME_REQUIREMENT[PostFunction.VIDEO]

        assert PostFunction.ZPROJECT in _MULTI_FRAME_REQUIREMENT
        assert "Z-slice" in _MULTI_FRAME_REQUIREMENT[PostFunction.ZPROJECT]

        assert PostFunction.COMPOSITE in _MULTI_FRAME_REQUIREMENT
        assert "channel" in _MULTI_FRAME_REQUIREMENT[PostFunction.COMPOSITE]

        assert PostFunction.STITCHED in _MULTI_FRAME_REQUIREMENT
        assert "tile" in _MULTI_FRAME_REQUIREMENT[PostFunction.STITCHED]

    def test_load_folder_empty_message_actionable(self):
        # When the folder truly contains no images: tell the user that's
        # what we checked for, and what to verify.
        src = (REPO_ROOT / "modules" / "protocol_post_processor.py").read_text()
        assert "No image files were found in the selected folder" in src
        assert "captured scan images" in src

    def test_helper_empty_message_actionable(self):
        src = (REPO_ROOT / "modules" / "protocol_post_processing_helper.py").read_text()
        assert "captured scan images" in src

    def test_composite_no_channels_message_specific(self):
        src = (REPO_ROOT / "modules" / "composite_generation.py").read_text()
        assert "no channel images" in src
        # The old generic wording must not survive a future merge.
        assert 'Composite Generation Error: No images found"' not in src

    def test_generic_no_images_found_wording_removed(self):
        # The bare "'No images found'" return message was the L1 reader's
        # complaint -- the folder had images, but the message said
        # otherwise. It should not be a literal string anywhere in the
        # post-processing files anymore.
        for rel in (
            "modules/protocol_post_processor.py",
            "modules/protocol_post_processing_helper.py",
        ):
            src = (REPO_ROOT / rel).read_text()
            assert "'No images found'" not in src, rel
            assert "'No images found in selected folder'" not in src, rel


class TestFolderPickerNative:
    """All FolderChooseBTN contexts use the OS-native folder picker.

    Per #675 (broader revert): the in-app Kivy picker introduced for
    post-processing contexts was removed -- native OS pickers on every
    supported platform already show file listings, so the duplicate UX
    surface didn't earn its keep.
    """

    def test_macos_choose_folder_helper_exists(self):
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        assert "def _macos_choose_folder(" in src
        # AppleScript "choose folder" is the macOS native folder browser.
        assert "choose folder" in src

    def test_platform_native_helper_dispatches_by_os(self):
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        assert "def _platform_native_choose_folder(" in src
        # macOS branch
        assert "sys.platform == 'darwin'" in src
        # Windows/Linux branch (tkinter)
        assert "askdirectory" in src

    def test_folder_choose_btn_uses_native_picker_for_all_contexts(self):
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        # Carve out the choose() method body specifically -- the
        # on_selection_function() method legitimately still branches on
        # self.context == 'live_folder' to decide WHERE to write the
        # result, which is orthogonal to WHICH picker to open.
        after_class = src.split("class FolderChooseBTN")[1]
        choose_start = after_class.find("def choose(")
        choose_after = after_class[choose_start:]
        choose_body = choose_after.split("\n    def ")[0]
        # The native helper is the single canonical call.
        assert "_platform_native_choose_folder" in choose_body
        # And the now-retired Kivy picker call must not reappear.
        assert "_open_kivy_folder_picker" not in choose_body
        # No per-context conditional remains that gates picker choice.
        assert "self.context == 'live_folder'" not in choose_body
