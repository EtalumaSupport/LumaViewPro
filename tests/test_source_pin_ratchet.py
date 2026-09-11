# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Ratchet on tests that pin production SOURCE TEXT.

A test that asserts on source text passes or fails on formatting: `'def
x(' in src` breaks when the signature is wrapped or gains a parameter,
and keeps passing when the function is gutted. The right shape asserts
the seam or the behaviour:

  * a def / method / signature      -> `tests.ast_seams.assert_def`
  * a name that must NOT appear      -> walk `tests.ast_seams.parse_module`
  * a Kivy handler's behaviour       -> call it UNBOUND (recipe below)
  * a .kv rule, a doc example, a
    comment's wording, a file the
    run wrote                        -> read it, with `# pin-justified:`

Every `read_text` call in tests/ is classified by what it reads
(`classify_read_text_sites`), and every bucket is announced at the end
of each run. ONE bucket is pinned: `body-pin`, a literal `.py` path read
with no justification beside it -- the fragile shape. The pin is an
EQUALITY per file, like the GUI display-only pins: a rise names the
remedy, a fall lowers the file's entry in the same commit, and the pin
is never raised. The other buckets (a path from a variable, a scan over
many files, a data or doc read, a `.kv` read, a justified `.py` read)
are legitimate and unpinned, so a manifest read-back or an AST parse
never collides with this guard.

Justifying a `.py` read: put `# pin-justified: <why there is no seam>`
on the read's line or within the three lines above it. The
justification lives at the site, self-describing, not in a ledger here.

Kivy handlers are testable without a Window. Call the handler UNBOUND on
a `types.SimpleNamespace` that carries only the attributes it touches
(`ModSlider.on_touch_down(fake_self, touch)`); monkeypatch the module's
`_app_ctx.ctx` to a fake ctx whose panels' `collide_point` returns
False; set `Window.modifiers` on the conftest's MagicMock `Window` with
`monkeypatch.setattr(mod.Window, 'modifiers', [...], raising=False)`.
Widgets cannot be INSTANTIATED under the stubbed base (no
`register_event_type`); the unbound call sidesteps that entirely, and a
mutation of the handler's branch turns exactly the test for that branch
red.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, iter_package_modules

_JUSTIFICATION = 'pin-justified'
_JUSTIFICATION_WINDOW = 3  # lines above the read that may carry it


def _nodes_inside_iteration(tree):
    """ids of nodes lexically inside a for-loop or comprehension."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            for sub in ast.walk(node):
                inside.add(id(sub))
    return inside


def _is_justified(lines, lineno):
    window = lines[max(0, lineno - 1 - _JUSTIFICATION_WINDOW) : lineno]
    return any(_JUSTIFICATION in line for line in window)


def classify_read_text_sites():
    """Group every `read_text` call in tests/ by what it is doing.

    Returns a dict of category -> list of 'file:line'. Categories carry
    different dispositions, which is the whole reason to separate them:
    a hygiene scan should never migrate, a `.kv` pin cannot, a justified
    `.py` read has said why, and an unjustified `body-pin` should.
    """
    found: dict[str, list[str]] = {
        'body-pin': [],
        'body-pin-justified': [],
        'single-computed': [],
        'hygiene-scan': [],
        'data-doc-read': [],
        'kv-pin': [],
    }
    for rel_path, tree in iter_package_modules(('tests',)):
        in_loop = _nodes_inside_iteration(tree)
        lines = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != 'read_text':
                continue
            expr = ast.unparse(node)
            if id(node) in in_loop:
                category = 'hygiene-scan'
            elif '.kv' in expr:
                category = 'kv-pin'
            elif '.py' in expr:
                if lines is None:
                    lines = (REPO_ROOT / rel_path).read_text().splitlines()
                category = 'body-pin-justified' if _is_justified(lines, node.lineno) else 'body-pin'
            elif any(ext in expr for ext in ('.md', '.json', '.txt')):
                category = 'data-doc-read'
            else:
                category = 'single-computed'
            found[category].append(f'{rel_path}:{node.lineno}')
    return found


def _fragile_pin_counts():
    """{'tests/<file>.py': number of unjustified literal-.py reads}."""
    counts: dict[str, int] = {}
    for entry in classify_read_text_sites()['body-pin']:
        rel_path = entry.rsplit(':', 1)[0]
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


# Pinned at 132d09a9 (beta35). Lower a value in the same commit that moves
# the assertion to a seam or justifies the read; never raise one.
_FRAGILE_PIN = {
    'tests/test_audit_fixes.py': 75,
    'tests/test_camera_log_routing.py': 1,
    'tests/test_camera_sdk_probe_observability.py': 2,
    # 4 -> 3: test_reconnect_regates_the_ui was deleted with the dead
    # MicroscopeSettings.reconnect handler it pinned, and its assertion
    # read ui/microscope_settings.py as source text. Nothing replaced it;
    # the invariant went with the handler.
    'tests/test_capability_gating_ssot.py': 3,
    'tests/test_composite_channel_extract_672.py': 2,
    'tests/test_controls_lockout.py': 5,
    'tests/test_dark_floor_capture_guard.py': 1,
    'tests/test_enhance_file_or_folder.py': 7,
    'tests/test_fatal_abort_led_safety.py': 2,
    'tests/test_histogram_display_gating.py': 1,
    'tests/test_image_mode.py': 2,
    'tests/test_init_z_sync.py': 1,
    'tests/test_installer_log_capture.py': 1,
    'tests/test_issue_568_protocol_time_floor.py': 1,
    'tests/test_issue_629_zproj_picker.py': 1,
    'tests/test_issue_655_ag_ae_exposure_cap.py': 1,
    'tests/test_issue_684_jpg_quality_row.py': 1,
    'tests/test_issue_691_video_duration.py': 1,
    'tests/test_issue_697_nav_led_sweep.py': 3,
    'tests/test_issue_749_color2ch_contract.py': 1,
    'tests/test_least_astonishment_fixes.py': 7,
    'tests/test_led_ack_and_tsr_filename.py': 1,
    'tests/test_logger_bundle_single_owner.py': 1,
    'tests/test_manual_recording_controller.py': 1,
    'tests/test_periodic_current_json_flush.py': 1,
    'tests/test_popup_close_button.py': 2,
    'tests/test_post_processing_time_trendline.py': 1,
    'tests/test_protocol_modules.py': 1,
    'tests/test_protocol_move_io_ordering.py': 1,
    'tests/test_quick_enhance.py': 2,
    'tests/test_quick_enhance_kv.py': 1,
    'tests/test_record_path_buffer_reuse.py': 1,
    'tests/test_root_logging_capture.py': 1,
    'tests/test_run_encoding_ssot.py': 1,
    'tests/test_single_instance_popup_focus.py': 1,
    'tests/test_stitcher.py': 4,
    'tests/test_tsr_cluster_fix.py': 3,
}

_REMEDY = (
    'A test is pinning production source text. Assert the seam or the '
    'behaviour instead: tests.ast_seams.assert_def for a def; walk '
    'tests.ast_seams.parse_module for a name that must be absent; for a '
    'Kivy handler, call it unbound on a SimpleNamespace (recipe in this '
    "module's docstring). If the subject truly has no seam, put "
    '"# pin-justified: <why>" within three lines above the read. '
    'This pin is never raised.'
)


def _ratchet_report(pin, actual):
    lines = []
    for key in sorted(set(pin) | set(actual)):
        before, now = pin.get(key, 0), actual.get(key, 0)
        if now > before:
            lines.append(f'{key}: fragile source pins rose {before} -> {now}. {_REMEDY}')
        elif now < before:
            lines.append(
                f'{key}: fragile source pins fell {before} -> {now}. '
                f'Lower _FRAGILE_PIN in this commit.'
            )
    return lines


def test_fragile_source_pins_match_the_pin():
    """New tests assert seams or behaviour, not source text; every drop is
    recorded where it happened."""
    report = _ratchet_report(_FRAGILE_PIN, _fragile_pin_counts())
    assert report == [], '\n'.join(report)


def test_classification_covers_every_site():
    """The published split must account for every site.

    Without this the announced buckets could drift from reality while the
    pin above still passes -- a classification nobody can trust is worse
    than no classification, because the migration is scoped from it.
    """
    sites = classify_read_text_sites()
    total = sum(len(v) for v in sites.values())
    assert total > 0, 'the scan found nothing, so it is no longer scanning'
    for category, entries in sites.items():
        assert all(':' in entry for entry in entries), f'{category} has a malformed entry'


def test_justification_is_read_from_the_site():
    """A `# pin-justified:` comment within the window moves a `.py` read
    out of the fragile bucket; one line past the window does not."""
    src = (
        'from pathlib import Path\n'
        '# pin-justified: an example with no seam\n'
        "a = Path('x.py').read_text()\n"
        '\n'
        '# pin-justified: too far above\n'
        '\n'
        '\n'
        '\n'
        "b = Path('y.py').read_text()\n"
    )
    lines = src.splitlines()
    assert _is_justified(lines, 3)
    assert not _is_justified(lines, 9)


# Announced at the end of every run (tests/ratchets.py).
from tests import ratchets as _ratchets

_ratchets.register(
    'tests: fragile source pins (unjustified literal-.py reads)',
    lambda: sum(_fragile_pin_counts().values()),
    sum(_FRAGILE_PIN.values()),
    'equal',
)
for _bucket in ('body-pin-justified', 'single-computed', 'hygiene-scan', 'data-doc-read', 'kv-pin'):
    _ratchets.register(
        f'tests: read_text sites, {_bucket}',
        (lambda b=_bucket: len(classify_read_text_sites()[b])),
        0,
        'announce',
    )
