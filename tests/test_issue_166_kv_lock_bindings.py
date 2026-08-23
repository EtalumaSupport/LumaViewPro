# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#166 regression: interactive controls lock out while a protocol runs.

An app-level ``run_lockout`` BooleanProperty mirrors the session's
``run_lockout`` derivation. The kv binds ``disabled: app.run_lockout``
on every LOCK control so the whole control surface greys out during a scan --
EXCEPT a small KEEP roster (the two view-transform buttons, the three
run/abort toggles, the panel show/hide toggles, the read-only support/log
buttons, and the display-only histogram log toggle).

This test reads ``ui/lumaviewpro.kv`` as text (the suite mocks Kivy; it does
not instantiate widgets) and asserts the binding structure:

- POSITIVE: a representative LOCK control from every region is disabled during
  a protocol -- either its own widget block carries ``disabled:
  app.run_lockout`` OR an ancestor container (a ``<Class>:`` rule root)
  does. The OR-combined controls (obj_position, Save, quality_stitch_btn,
  fast_preview_stitch_btn) carry
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

BIND = 'disabled: app.run_lockout'
BIND_NOSPACE = 'disabled:app.run_lockout'


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
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
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
    for line in lines[header + 1 :]:
        if line.strip() == '':
            block.append(line)
            continue
        if _indent(line) <= header_indent:
            break
        block.append(line)
    return block


def _block_has_bind(block: list[str]) -> bool:
    # controls_locked derives from run_lockout OR recording_active,
    # so either token satisfies "disabled during a protocol".
    for line in block:
        if line.strip().startswith('disabled:') and (
            'app.run_lockout' in line or 'app.controls_locked' in line
        ):
            return True
    return False


def _ancestor_has_bind(control_id: str) -> bool:
    """True when any enclosing widget block (within the kv rule) binds
    disabled to the protocol lock. Container binds moved off the rule
    roots and onto interior containers when the stop toggles were
    exempted; the cascade locks children the same way from there."""
    lines = _kv_lines()
    id_pat = re.compile(r'^[ \t]*id:\s*' + re.escape(control_id) + r'\s*(#.*)?$')
    id_idx = next(i for i, line in enumerate(lines) if id_pat.match(line))
    depth = _indent(lines[id_idx])
    j = id_idx - 1
    while j >= 0 and not lines[j].startswith('<'):
        line = lines[j]
        if line.strip() and not line.lstrip().startswith('#') and _indent(line) < depth:
            # An ancestor header: scan its OWN property lines only. In kv,
            # properties and child widgets share the same indent level, so
            # the scan stops at the first child widget header (a
            # capitalised `ClassName:` line) -- a bind inside a child
            # subtree is not the ancestor's.
            header_indent = _indent(line)
            child_header = re.compile(r'^[ \t]+[A-Z][A-Za-z_]*:\s*(#.*)?$')
            for k in range(j + 1, id_idx + 1):
                prop = lines[k]
                if not prop.strip() or prop.lstrip().startswith('#'):
                    continue
                if _indent(prop) <= header_indent or child_header.match(prop):
                    break
                if prop.strip().startswith('disabled:') and (
                    'app.run_lockout' in prop or 'app.controls_locked' in prop
                ):
                    return True
            depth = header_indent
        j -= 1
    return False


def _class_rule_root_has_bind(class_name: str) -> bool:
    """True if the ``<class_name>:`` rule binds disabled at its rule root.

    A rule-root property is one indented exactly one level under the
    ``<Class>:`` header (i.e. deeper than the header, shallower than any
    nested child widget's properties). We accept any ``disabled: ...
    app.run_lockout`` line that sits at the shallowest property depth
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
    for line in lines[start + 1 :]:
        if line.strip() == '' or line.lstrip().startswith('#'):
            continue
        if _indent(line) <= header_indent:
            break  # end of rule
        # First non-comment property line establishes the rule-root depth.
        if line.strip().startswith('disabled:') and (
            'app.run_lockout' in line or 'app.controls_locked' in line
        ):
            return True
    return False


# --------------------------------------------------------------------------
# POSITIVE: representative LOCK controls (one per region) are disabled during
# a protocol -- via own block or an ancestor container rule.
# --------------------------------------------------------------------------

# control_id -> container <Class>: rule that container-binds it (or None when
# the control carries its own per-widget bind).
LOCK_REPRESENTATIVES = {
    # Focus / Z jog + turret/objective moved off the <VerticalControl>
    # rule root onto interior containers when the stop toggles were
    # exempted -- 'ANCESTOR' walks the enclosing blocks instead.
    # zstack_aqr_btn left this table for TestStopToggleExemption: it is
    # a stop-capable toggle, deliberately NOT protocol-locked.
    'fast_up': 'ANCESTOR',
    'obj_position': 'ANCESTOR',
    'objective_spinner2': 'ANCESTOR',
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
    'quality_stitch_btn': None,
    'fast_preview_stitch_btn': None,
    # Microscope settings (mixed region -> per-widget).
    'image_mode_spinner': None,
    'binning_spinner': None,
    'enable_crosshairs_btn': None,
}


def test_representative_lock_controls_disabled_during_protocol():
    for control_id, container in LOCK_REPRESENTATIVES.items():
        block = _block_for_id(control_id)
        own = _block_has_bind(block)
        if container == 'ANCESTOR':
            ancestor = _ancestor_has_bind(control_id)
        else:
            ancestor = bool(container) and _class_rule_root_has_bind(container)
        assert own or ancestor, (
            f'LOCK control {control_id!r} must be disabled during a protocol: '
            f'expected a protocol-lock bind in its own widget block, an '
            f'enclosing container block, or its container rule {container!r}. '
            f'(#166)'
        )


STOP_TOGGLES = (
    'run_autofocus_btn',
    'run_scan_btn',
    'run_protocol_btn',
    'autofocus_id',
    'zstack_aqr_btn',
)


class TestStopToggleExemption:
    """A run's own toggle IS its stop control: it must stay clickable
    during the run (second click = abort) and lock only while a rival
    RECORDING is live. A protocol-lock bind on the toggle or any
    ancestor strands the abort -- the exact regression the #166 KEEP
    roster warned about, reintroduced from above by a container bind."""

    def test_stop_toggles_lock_only_for_recordings(self):
        for control_id in STOP_TOGGLES:
            block = _block_for_id(control_id)
            assert any(
                line.strip().startswith('disabled:') and 'app.recording_active' in line
                for line in block
            ), f'{control_id!r} must carry disabled: app.recording_active'

    def test_stop_toggles_have_no_protocol_locked_ancestor(self):
        for control_id in STOP_TOGGLES:
            assert not _ancestor_has_bind(control_id), (
                f'{control_id!r} sits under a protocol-locked container; '
                f'its abort click would be swallowed mid-run'
            )
            assert not _block_has_bind(_block_for_id(control_id)), (
                f'{control_id!r} must not carry a protocol-lock bind itself'
            )


def test_container_bound_regions_carry_root_bind():
    """The all-LOCK regions are locked once at their rule root."""
    for class_name in (
        # VerticalControl left this list when the stop toggles were
        # exempted: its lock moved to interior containers (see
        # TestStopToggleExemption + the ANCESTOR representatives).
        'XYStageControl',  # XY stage jog
        'VideoCreationControls',  # post-processing: video
        'ZProjectionControls',  # post-processing: z-projection
        'CompositeGenControls',  # post-processing: composite
        'GraphingControls',  # graphing popup
        'CellCountControls',  # object-analysis popup
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
# app.run_lockout in that line (never overwritten away).
# --------------------------------------------------------------------------


def test_or_combined_controls_keep_run_lockout():
    # The stitch buttons shipped `disabled: False`; the placeholder is
    # replaced with the property (False OR x == x). obj_position's own
    # bind retired when its container took the lock (exemption work).
    for control_id in ('quality_stitch_btn', 'fast_preview_stitch_btn'):
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
    window = lines[max(0, save_idx - 10) : save_idx + 2]
    assert any(
        ln.strip().startswith('disabled:') and 'app.run_lockout' in ln and _indent(ln) == base
        for ln in window
    ), 'Save protocol button must carry disabled: app.run_lockout (#166)'


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
    # Read-only OS folder open -- must stay reachable mid-run.
    'open_last_save_folder',
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
            f'protocol; it must not be disabled by app.run_lockout. (#166)'
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
