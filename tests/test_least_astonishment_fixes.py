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
        src = (REPO_ROOT / 'modules' / 'config_helpers.py').read_text()
        # Quote-style agnostic: ruff format may use single or double
        # quotes for the f-string.
        m = re.search(
            r"""futures_parts\.append\(\s*f["'](?P<fmt>[^"']+)["']\s*\)""",
            src,
        )
        assert m is not None, 'futures_parts append line not found'
        fmt = m.group('fmt')
        assert 'alloc=' in fmt, f'alloc= missing from: {fmt!r}'
        assert 'pop=' in fmt, f'pop= missing from: {fmt!r}'
        assert 'live=' in fmt, f'live= missing from: {fmt!r}'

    def test_old_abbreviated_format_not_present(self):
        src = (REPO_ROOT / 'modules' / 'config_helpers.py').read_text()
        # The old format invited Active/Pending/Leaked misread; must not
        # come back in any form.
        assert '=A{' not in src
        assert '/P{' not in src
        assert '/L{' not in src


class TestPostProcessingEmptyResultMessages:
    """Empty-result popups name the missing capture dimension per PostFunction."""

    def test_multi_frame_requirement_covers_each_post_function(self):
        from modules.common_utils import PostFunction
        from modules.protocol_post_processor import _MULTI_FRAME_REQUIREMENT

        assert PostFunction.VIDEO in _MULTI_FRAME_REQUIREMENT
        assert 'time point' in _MULTI_FRAME_REQUIREMENT[PostFunction.VIDEO]

        assert PostFunction.ZPROJECT in _MULTI_FRAME_REQUIREMENT
        assert 'Z-slice' in _MULTI_FRAME_REQUIREMENT[PostFunction.ZPROJECT]

        assert PostFunction.COMPOSITE in _MULTI_FRAME_REQUIREMENT
        assert 'channel' in _MULTI_FRAME_REQUIREMENT[PostFunction.COMPOSITE]

        assert PostFunction.STITCHED in _MULTI_FRAME_REQUIREMENT
        assert 'tile' in _MULTI_FRAME_REQUIREMENT[PostFunction.STITCHED]

    def test_load_folder_empty_message_actionable(self):
        # When the folder truly contains no images: tell the user that's
        # what we checked for, and what to verify.
        # pin-justified: the user-facing message wording is the contract.
        src = (REPO_ROOT / 'modules' / 'protocol_post_processor.py').read_text()
        assert 'No image files were found in the selected folder' in src
        assert 'captured scan images' in src

    def test_helper_empty_message_actionable(self):
        # pin-justified: the user-facing message wording is the contract.
        src = (REPO_ROOT / 'modules' / 'protocol_post_processing_helper.py').read_text()
        assert 'captured scan images' in src

    def test_composite_no_channels_message_specific(self):
        # pin-justified: the user-facing message wording is the contract.
        src = (REPO_ROOT / 'modules' / 'composite_generation.py').read_text()
        assert 'no channel images' in src
        # The old generic wording must not survive a future merge.
        assert 'Composite Generation Error: No images found"' not in src

    def test_generic_no_images_found_wording_removed(self):
        # The bare "'No images found'" return message was the L1 reader's
        # complaint -- the folder had images, but the message said
        # otherwise. It should not be a literal string anywhere in the
        # post-processing files anymore.
        for rel in (
            'modules/protocol_post_processor.py',
            'modules/protocol_post_processing_helper.py',
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
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        assert 'def _macos_choose_folder(' in src
        # AppleScript "choose folder" is the macOS native folder browser.
        assert 'choose folder' in src

    def test_platform_native_helper_dispatches_by_os(self):
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        assert 'def _platform_native_choose_folder(' in src
        # macOS branch
        assert "sys.platform == 'darwin'" in src
        # Windows/Linux branch (tkinter)
        assert 'askdirectory' in src

    def test_folder_choose_btn_uses_native_picker_for_all_contexts(self):
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        # Carve out the choose() method body specifically -- the
        # on_selection_function() method legitimately still branches on
        # self.context == 'live_folder' to decide WHERE to write the
        # result, which is orthogonal to WHICH picker to open.
        after_class = src.split('class FolderChooseBTN')[1]
        choose_start = after_class.find('def choose(')
        choose_after = after_class[choose_start:]
        choose_body = choose_after.split('\n    def ')[0]
        # The native helper is the single canonical call.
        assert '_platform_native_choose_folder' in choose_body
        # And the now-retired Kivy picker call must not reappear.
        assert '_open_kivy_folder_picker' not in choose_body
        # No per-context conditional remains that gates picker choice.
        assert "self.context == 'live_folder'" not in choose_body


class TestZprojectionFolderPickerDefaultDepth_629:
    """Issue #629 v2: the apply_zprojection_to_folder picker must
    default to the deepest existing canonical Z-stack tree, not the
    grandparent. The manual ZSTACK button writes into
    live_folder/Manual/Z-Stacks/<ts>/ (ui/zstack.py:234); the picker
    should land there so each timestamped run is one click away.

    Prior fix (e365865) moved the default UP from live_folder/
    ProtocolData/ to live_folder/, fixing the "one level too deep"
    complaint but creating a "two levels too high" regression: the
    user landed at the parent of Manual/ and ProtocolData/ and had to
    drill down through both before reaching the actual run folders.

    Fix: extract the default-path logic into a pure helper
    _zprojection_picker_default_path(live_folder) that returns the
    most-specific existing path among
    (Manual/Z-Stacks, ProtocolData, live_folder). The picker branch
    in FolderChooseBTN.choose() delegates to the helper.

    Tests use the source-text pattern of the sibling
    TestFolderPickerNative class so they don't drag in Kivy imports.
    """

    def test_helper_function_exists_with_pure_signature(self):
        """Helper must exist at module scope, take live_folder, return
        a str. Pure (no Kivy / no app_ctx / no settings access). Future
        behavioral tests can import it directly without conftest changes."""
        from tests.ast_seams import assert_def

        assert_def(
            'ui/file_dialogs.py', '_zprojection_picker_default_path',
            has_params=['live_folder'],
            msg='Helper _zprojection_picker_default_path() must exist; '
                'FolderChooseBTN.choose() delegates to it (#629).',
        )

    def test_helper_descends_manual_z_stacks_first(self):
        """Helper body must search Manual/Z-Stacks BEFORE ProtocolData.
        Reversed priority would land the user inside ProtocolData
        first, regressing back to the original symptom."""
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        start = src.find('def _zprojection_picker_default_path(')
        assert start != -1
        body_section = src[start : start + 1500]
        body_end = body_section.find('\ndef ', 1)
        if body_end == -1:
            body_end = body_section.find('\nclass ', 1)
        if body_end != -1:
            body_section = body_section[:body_end]
        # Quote-style agnostic: ruff format may use single or double quotes.
        import re

        manual_match = re.search(r"""["']Manual["']""", body_section)
        protocol_match = re.search(r"""["']ProtocolData["']""", body_section)
        assert manual_match is not None, 'helper must include Manual subpath (#629)'
        assert protocol_match is not None, 'helper must include ProtocolData fallback (#629)'
        manual_idx = manual_match.start()
        protocol_idx = protocol_match.start()
        assert manual_idx < protocol_idx, (
            'Manual/Z-Stacks must come BEFORE ProtocolData in the '
            'candidate priority order so manual z-stack workflow is '
            'one click away (#629). Reversed order regresses to the '
            "original 'one level too deep' symptom."
        )

    def test_helper_falls_back_to_live_folder(self):
        """Helper must contain a final fallback that returns the
        live_folder itself, so a fresh install with neither subtree
        present still produces a valid picker target."""
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        start = src.find('def _zprojection_picker_default_path(')
        body_section = src[start : start + 1500]
        body_end = body_section.find('\ndef ', 1)
        if body_end == -1:
            body_end = body_section.find('\nclass ', 1)
        if body_end != -1:
            body_section = body_section[:body_end]
        # The fallback after the candidate loop must return the base
        # path itself; "return str(base)" is the canonical phrasing.
        assert 'return str(base)' in body_section, (
            'helper must fall back to live_folder when neither subtree '
            'exists (#629). Without the fallback, a fresh install '
            'yields no valid picker target.'
        )

    def test_choose_method_delegates_to_helper_for_zprojection(self):
        """FolderChooseBTN.choose() apply_zprojection_to_folder branch
        must invoke the helper. Inline path logic in the branch is the
        shape that produced the over-correction; the helper centralizes
        the priority order in one testable place."""
        src = (REPO_ROOT / 'ui' / 'file_dialogs.py').read_text()
        after_class = src.split('class FolderChooseBTN')[1]
        choose_start = after_class.find('def choose(')
        choose_body = after_class[choose_start:].split('\n    def ')[0]
        zproject_idx = choose_body.find('apply_zprojection_to_folder')
        assert zproject_idx != -1, 'choose() must have apply_zprojection_to_folder branch'
        zproject_branch = choose_body[zproject_idx : zproject_idx + 800]
        assert '_zprojection_picker_default_path(' in zproject_branch, (
            'FolderChooseBTN.choose() apply_zprojection_to_folder '
            'branch must delegate to _zprojection_picker_default_path() '
            '(#629). Inline path logic in the branch was prone to '
            'over-correction by ad-hoc patches.'
        )
