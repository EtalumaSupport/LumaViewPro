# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#166 regression: interactive controls lock out while a protocol runs.

An app-level ``protocol_running`` BooleanProperty mirrors the worker-thread
``ctx.protocol_running`` Event. The kv binds ``disabled: app.protocol_running``
on every LOCK control so the whole control surface greys out during a scan --
EXCEPT a small KEEP roster (the two view-transform buttons, the three
run/abort toggles, the panel show/hide toggles, the read-only support/log
buttons, and the display-only histogram log toggle).

This test reads ``ui/lumaviewpro.kv`` as text (the suite mocks Kivy; it does
not instantiate widgets) and asserts the binding structure:

- POSITIVE: a representative LOCK control from every region is disabled during
  a protocol -- either its own widget block carries ``disabled:
  app.protocol_running`` OR an ancestor container (a ``<Class>:`` rule root)
  does. The OR-combined controls (obj_position, Save, stitch_apply_btn) carry
  it on their own block even though another token may be present.
- NEGATIVE (critical): the KEEP roster -- fit_btn, one2one_btn,
  show_tooltips_btn, btn_support_report, btn_zip_logs, logHistogram_id,
  toggle_motionsettings, toggle_imagesettings, and the three run/abort toggles
  -- must NOT carry the binding in their own widget block. Stranding the abort
  toggle is the worst regression this guards against.

Whitespace tolerant: the kv mixes tabs and spaces, so all comparisons
normalise leading indentation to a depth count and strip inner whitespace.
"""

from __future__ import annotations

import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
KV = REPO / 'ui' / 'lumaviewpro.kv'

BIND = 'disabled: app.protocol_running'
BIND_NOSPACE = 'disabled:app.protocol_running'


def _kv_lines() -> list[str]:
    # pin-justified: kv is declarative source with no headless seam; the
    # per-widget disabled binds in the kv text are the lockout contract.
    return KV.read_text().splitlines()


def _indent(line: str) -> int:
    """Indentation width, mirroring Kivy's parser.

    Kivy's kv parser replaces each tab with 4 spaces before measuring
    indentation (see kivy/lang/parser.py). The kv file mixes tabs and spaces
    within a block, so we must expand tabs the same way to get the depths
    Kivy actually sees -- a raw character count misreads tab/space mixes.
    """
    prefix = line[:len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _block_for_id(control_id: str) -> list[str]:
    """Return the full widget block that contains ``id: <control_id>``.

    In kv a widget's properties (``id:``, ``disabled:``, ``on_release:`` ...)
    are siblings at the same indentation, one level deeper than the widget's
    header line. So the block is anchored to the HEADER: walk up from the
    ``id:`` line to the nearest line at a shallower indent (the widget
    header), then forward to the line before the next sibling/outdent at the
    header's indentation. That span is exactly the one widget.
    """
    lines = _kv_lines()
    id_pat = re.compile(r'^[ \t]*id:\s*' + re.escape(control_id) + r'\s*(#.*)?$')
    id_idx = None
    for i, line in enumerate(lines):
        if id_pat.match(line):
            assert id_idx is None, f'id: {control_id} appears more than once'
            id_idx = i
    assert id_idx is not None, f'id: {control_id} not found in {KV.name}'

    prop_indent = _indent(lines[id_idx])

    # Walk up to the widget header (nearest prior non-blank line shallower
    # than the property indentation).
    header = id_idx
    j = id_idx - 1
    while j >= 0:
        if lines[j].strip() == '' or lines[j].lstrip().startswith('#'):
            j -= 1
            continue
        if _indent(lines[j]) < prop_indent:
            header = j
            break
        j -= 1
    header_indent = _indent(lines[header])

    block = [lines[header]]
    for line in lines[header + 1:]:
        if line.strip() == '':
            block.append(line)
            continue
        if _indent(line) <= header_indent:
            break
        block.append(line)
    return block


def _block_has_bind(block: list[str]) -> bool:
    for line in block:
        if line.strip().startswith('disabled:') and 'app.protocol_running' in line:
            return True
    return False


def _class_rule_root_has_bind(class_name: str) -> bool:
    """True if the ``<class_name>:`` rule binds disabled at its rule root.

    A rule-root property is one indented exactly one level under the
    ``<Class>:`` header (i.e. deeper than the header, shallower than any
    nested child widget's properties). We accept any ``disabled: ...
    app.protocol_running`` line that sits at the shallowest property depth
    of the rule before the first nested widget.
    """
    lines = _kv_lines()
    header = '<' + class_name + '>:'
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    assert start is not None, f'rule {header} not found'

    header_indent = _indent(lines[start])
    for line in lines[start + 1:]:
        if line.strip() == '' or line.lstrip().startswith('#'):
            continue
        if _indent(line) <= header_indent:
            break  # end of rule
        # First non-comment property line establishes the rule-root depth.
        if line.strip().startswith('disabled:') and 'app.protocol_running' in line:
            return True
    return False


# --------------------------------------------------------------------------
# POSITIVE: representative LOCK controls (one per region) are disabled during
# a protocol -- via own block or an ancestor container rule.
# --------------------------------------------------------------------------

# control_id -> container <Class>: rule that container-binds it (or None when
# the control carries its own per-widget bind).
LOCK_REPRESENTATIVES = {
    # Focus / Z jog + turret/objective + z-stack are under <VerticalControl>.
    'fast_up': 'VerticalControl',
    'obj_position': None,      # OR-combined on its own block
    'objective_spinner2': 'VerticalControl',
    'zstack_aqr_btn': 'VerticalControl',
    # Protocol edit / step-nav (mixed region -> per-widget).
    'capture_period': None,
    'labware_spinner': None,
    'add_step_btn': None,
    # Layer control (mixed region -> per-widget).
    'ill_slider': None,
    'gain_slider': None,
    'acquire_image': None,
    'false_color': None,
    # Post-processing (stitch is per-widget OR-combine; others container-bound).
    'stitch_apply_btn': None,
    # Microscope settings (mixed region -> per-widget).
    'enable_full_pixel_depth_btn': None,
    'binning_spinner': None,
    'enable_crosshairs_btn': None,
}


def test_representative_lock_controls_disabled_during_protocol():
    for control_id, container in LOCK_REPRESENTATIVES.items():
        block = _block_for_id(control_id)
        own = _block_has_bind(block)
        ancestor = bool(container) and _class_rule_root_has_bind(container)
        assert own or ancestor, (
            f'LOCK control {control_id!r} must be disabled during a protocol: '
            f'expected {BIND!r} in its own widget block or on its container '
            f'rule {container!r}. (#166)'
        )


def test_container_bound_regions_carry_root_bind():
    """The all-LOCK regions are locked once at their rule root."""
    for class_name in (
        'VerticalControl',        # focus / turret / objective / z-stack
        'XYStageControl',         # XY stage jog
        'VideoCreationControls',  # post-processing: video
        'ZProjectionControls',    # post-processing: z-projection
        'CompositeGenControls',   # post-processing: composite
        'GraphingControls',       # graphing popup
        'CellCountControls',      # object-analysis popup
    ):
        assert _class_rule_root_has_bind(class_name), (
            f'all-LOCK region <{class_name}>: must carry {BIND!r} at its rule '
            f'root so every child control disables during a protocol. (#166)'
        )


def test_xy_stage_control_locked_via_container():
    """home_id (XY) sits in the container-bound <XYStageControl> region."""
    assert _class_rule_root_has_bind('XYStageControl')


# --------------------------------------------------------------------------
# OR-combine: controls that shipped with a `disabled:` line keep
# app.protocol_running in that line (never overwritten away).
# --------------------------------------------------------------------------

def test_or_combined_controls_keep_protocol_running():
    # obj_position and stitch_apply_btn shipped `disabled: False`; the
    # placeholder is replaced with the property (False OR x == x).
    for control_id in ('obj_position', 'stitch_apply_btn'):
        block = _block_for_id(control_id)
        assert _block_has_bind(block), (
            f'{control_id!r} shipped a disabled: placeholder; it must now '
            f'carry {BIND!r} on its own block (OR-combined, not dropped). '
            f'(#166 / design 4.b)'
        )


def test_save_protocol_button_disabled():
    """The Save (protocol) button shipped `disabled: False`; now app-bound.

    It has no id, so locate it by its `root.save_protocol()` handler and
    confirm its block carries the bind.
    """
    lines = _kv_lines()
    save_idx = None
    for i, line in enumerate(lines):
        if 'root.save_protocol()' in line:
            save_idx = i
            break
    assert save_idx is not None, 'Save protocol button not found'
    # Walk back to the enclosing widget header, then forward over its block.
    base = _indent(lines[save_idx])
    # The block's other properties share save_idx's indentation; scan a small
    # window around the handler for the bind at the same depth.
    window = lines[max(0, save_idx - 10):save_idx + 2]
    assert any(
        ln.strip().startswith('disabled:') and 'app.protocol_running' in ln
        and _indent(ln) == base
        for ln in window
    ), 'Save protocol button must carry disabled: app.protocol_running (#166)'


# --------------------------------------------------------------------------
# NEGATIVE: KEEP controls must NOT be self-bound. Guards against an
# over-eager sweep stranding the abort button or a view-only control.
# --------------------------------------------------------------------------

KEEP_CONTROLS = (
    'fit_btn',
    'one2one_btn',
    'show_tooltips_btn',
    'btn_support_report',
    'btn_zip_logs',
    'logHistogram_id',
    'toggle_motionsettings',
    'toggle_imagesettings',
    # The three run/abort toggles -- Python-managed; must stay live to abort.
    'run_autofocus_btn',
    'run_scan_btn',
    'run_protocol_btn',
)


def test_keep_controls_not_self_bound():
    for control_id in KEEP_CONTROLS:
        block = _block_for_id(control_id)
        assert not _block_has_bind(block), (
            f'KEEP control {control_id!r} must NOT carry {BIND!r} in its own '
            f'widget block -- binding it would strand a view-only control or, '
            f'worst case, the abort toggle. (#166)'
        )


def test_run_abort_toggles_live_for_abort():
    """The three run/abort toggles are the explicit Stop affordance.

    Belt-and-suspenders over test_keep_controls_not_self_bound: re-assert the
    abort toggles specifically, since stranding them is the single worst
    regression.
    """
    for control_id in ('run_autofocus_btn', 'run_scan_btn', 'run_protocol_btn'):
        block = _block_for_id(control_id)
        joined = '\n'.join(block)
        assert BIND not in joined and BIND_NOSPACE not in joined.replace(' ', ''), (
            f'{control_id!r} is an abort toggle and MUST remain live during a '
            f'protocol; it must not be disabled by app.protocol_running. (#166)'
        )


# --------------------------------------------------------------------------
# Count invariant: a representative-or-greater number of binds exists.
# --------------------------------------------------------------------------

def test_minimum_bind_count():
    """At least the 7 container binds + the named per-widget binds exist.

    A loose lower bound -- the real count is higher (~90) -- but enough to
    catch a regression that drops the whole sweep.
    """
    text = KV.read_text()
    count = text.count(BIND)
    assert count >= 50, (
        f'expected the protocol-lockout sweep to bind many controls; found '
        f'only {count} occurrences of {BIND!r}. (#166)'
    )
