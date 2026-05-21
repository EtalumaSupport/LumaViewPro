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


class TestFolderPickerReplacement:
    """Kivy folder picker is used for post-processing folder contexts.

    The image-save destination (live_folder) uses the OS-native picker --
    different UX need (user knows the target; doesn't need to see files
    inside). The TestFolderPickerLiveFolderNative class below covers
    that branch.
    """

    def test_folder_picker_popup_class_exists(self):
        # Kivy is mocked at test time (see conftest.py); the import alone
        # exercises the module's import-time wiring.
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        assert "class FolderPickerPopup(Popup):" in src
        assert "def _open_kivy_folder_picker(" in src
        # FileChooserListView is the widget that shows files and folders --
        # the core of the post-processing-picker fix.
        assert "FileChooserListView(" in src
        assert "dirselect=True" in src

    def test_folder_choose_btn_uses_kivy_picker_for_post_processing(self):
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        # Carve out FolderChooseBTN's class body so we don't pick up
        # FileChooseBTN's still-valid tkinter usage.
        after = src.split("class FolderChooseBTN")[1]
        choose_body = after.split("class ")[0]
        # Kivy picker is the default branch (everything except live_folder).
        assert "_open_kivy_folder_picker" in choose_body


class TestFolderPickerLiveFolderNative:
    """live_folder = image-save destination uses the OS-native folder picker.

    The post-processing contexts retain the in-app Kivy picker (different
    UX need -- inspect files in the candidate folder). live_folder is a
    pure folder-choice action where the OS-native browser is the right
    affordance.
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

    def test_folder_choose_btn_dispatches_native_for_live_folder(self):
        src = (REPO_ROOT / "ui" / "file_dialogs.py").read_text()
        after = src.split("class FolderChooseBTN")[1]
        choose_body = after.split("class ")[0]
        # The conditional that gates native-vs-kivy by context.
        assert "self.context == 'live_folder'" in choose_body
        # The native helper is what the live_folder branch calls.
        assert "_platform_native_choose_folder" in choose_body
