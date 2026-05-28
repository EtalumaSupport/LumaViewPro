#!/usr/bin/env python3
"""Mechanical pre-commit gate for CLAUDE.md Rules 24/27/28/42.

Run as a pre-commit hook (via tools/install_hooks.py) or directly:

    python tools/check_rules.py --staged
    python tools/check_rules.py --staged --all
    python tools/check_rules.py --paths drivers/foo.py drivers/bar.py

Default mode in --staged is diff-only: violations are reported only when
they appear on lines added by the staged commit. Pre-existing violations
in untouched lines are not blocked. Use --all to flag every violation in
every modified file (useful for cleanup sweeps).

Rules implemented:
    rule_24  -- ASCII-only over the full source text per CLAUDE.md spec
                ('every string ... every comment ... every docstring ...
                every identifier in .py / .c / .h / .kv / similar files').
                Source-text scan (not AST), so escape sequences like
                '\\r\\n' read as ASCII source even though the resolved
                string value contains U+000D. File-level exempt:
                test_check_rules*.py. Line-level exempt registry:
                _RULE_24_LINE_EXEMPT (load-bearing literals only).
                Same source-text scan applied to .kv files.
    rule_27a -- no `# TODO` / `# FIXME` / `# XXX` in source comments
    rule_27b -- no rule / audit / session / smoke / wave / phase IDs in comments
    rule_27d -- same patterns as 27b but applied to docstrings
    rule_28  -- no internal IDs in notifications.{level} string args
    rule_42  -- WARN on "healthy"/"fine"/"within range" in comments without a
                `PERFORMANCE_BUDGETS.md` cite in the same file

Severities: 'block' fails the commit (exit 1); 'warn' prints to stderr
but does not affect exit code. Rule 42 is the only WARN-severity check
today.

Exit code: 0 clean (warns allowed), 1 blocks present.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Violation:
    path: str
    line: int
    col: int
    rule: str
    message: str
    severity: str = 'block'  # 'block' = exit 1; 'warn' = stderr only

    def format(self) -> str:
        prefix = 'WARN ' if self.severity == 'warn' else ''
        return f'{prefix}{self.path}:{self.line}:{self.col} {self.rule}: {self.message}'


_NON_ASCII = re.compile(r'[^\x09\x0a\x20-\x7e]')

_TODO_PATTERN = re.compile(r'\b(TODO|FIXME|XXX)\b')

_COMMENT_ID_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bRule\s+\d+\b'), 'Rule N reference'),
    (re.compile(r'\baudit\s+[A-Z][A-Za-z0-9_-]*\b', re.IGNORECASE), 'audit reference'),
    # Snake-case audit-doc identifiers -- the form audit docs ship with
    # as filenames. Distinct from the natural-language pattern above
    # because the snake-case form has no space between the keyword and
    # the rest of the identifier.
    (re.compile(r'\bAUDIT_[A-Z][A-Z0-9_-]*\b'), 'AUDIT_* doc reference'),
    (re.compile(r'\bfix\s+#\d+\b', re.IGNORECASE), 'fix #N reference'),
    (re.compile(r'\bsession\s+\d+\b', re.IGNORECASE), 'session N reference'),
    (re.compile(r'\bSmoke\s+\d+\b'), 'Smoke N reference'),
    (re.compile(r'\bWave\s+\d+\b'), 'Wave N reference'),
    (re.compile(r'\bPhase\s+[A-Z]\b'), 'Phase X reference'),
)

_INTERNAL_ID_PATTERN = re.compile(r'\b(A\d+|LV-\d+|F\d+|M\d+|LVP-A-\d+|Rule\s+\d+)\b')

_RULE_42_TRIGGER = re.compile(r'\b(healthy|fine|within\s+range)\b', re.IGNORECASE)
_RULE_42_SUPPRESS = re.compile(r'PERFORMANCE_BUDGETS\.md')

_LOGGER_BASES = frozenset({'logger', '_log', '_cam_log', 'lvp_logger'})
_LOGGER_METHODS = frozenset({'info', 'warning', 'error', 'critical', 'debug', 'exception'})
_NOTIFICATIONS_BASES = frozenset({'notifications'})
_NOTIFICATIONS_METHODS = frozenset({'info', 'warning', 'error', 'critical'})
_PRINT_NAMES = frozenset({'print'})


def _is_notification_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        if fn.value.id in _NOTIFICATIONS_BASES and fn.attr in _NOTIFICATIONS_METHODS:
            return True
    return False


def _walk_excluding_calls(node: ast.AST):
    """Walk AST yielding `node` and its descendants, but skipping into Call subtrees.

    A nested Call inside an arg expression (e.g. `logger.info(f"x: {f()}")`)
    has its own argument scope and may itself be checked separately if it's
    a logger/notification call. We don't want strings inside its args to
    count toward the outer call's args.
    """
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            continue
        yield from _walk_excluding_calls(child)


def _arg_string_constants(call: ast.Call) -> list[tuple[str, int, int]]:
    """Return (string, line, col) for every str Constant reachable from
    `call`'s direct arguments (and any non-Call subtrees of those args)."""
    results: list[tuple[str, int, int]] = []
    arg_roots: list[ast.AST] = list(call.args)
    arg_roots.extend(kw.value for kw in call.keywords)
    for root in arg_roots:
        for child in _walk_excluding_calls(root):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                results.append((child.value, child.lineno, child.col_offset))
    return results


_RULE_24_LINE_EXEMPT: dict[str, frozenset[int]] = {
    # CLI progress-bar block glyphs (U+2588 / U+2591) -- functional
    # output; intentionally non-ASCII for terminal rendering.
    'modules/tech_support_report.py': frozenset({2475}),
    # Test asserts LumascopeSkills.md handles both U+2026 and '...';
    # the unicode literal IS the test fixture, replacing it tautologizes.
    'tests/test_audit_fixes.py': frozenset({9716}),
}


def _is_rule_24_exempt(path: str) -> bool:
    """File-level exempt: test files whose purpose is to construct
    synthetic non-ASCII fixtures for the rule_24 check itself. Pattern
    matches test_check_rules*.py (sibling variants like rule_24_kv).
    """
    norm = path.replace('\\', '/')
    basename = norm.rsplit('/', 1)[-1]
    return basename.startswith('test_check_rules')


def _rule_24_line_exempt(path: str, line: int) -> bool:
    norm = path.replace('\\', '/')
    return line in _RULE_24_LINE_EXEMPT.get(norm, frozenset())


def _check_rule_24(source: str, path: str) -> list[Violation]:
    """ASCII-only check over the full source text.

    Per CLAUDE.md Rule 24, every string + every comment + every
    docstring + every identifier in .py / .c / .h / .kv files must be
    ASCII (0x20-0x7E plus tab + newline). Source-text line scan; this
    correctly excludes escape sequences like '\\r\\n' (the source bytes
    are ASCII even though the resolved string value contains U+000D).

    Exemptions:
      - File-level for `test_check_rules*.py` (synthetic fixtures).
      - Line-level via `_RULE_24_LINE_EXEMPT` (load-bearing literals).
    """
    if _is_rule_24_exempt(path):
        return []
    violations: list[Violation] = []
    for ln, line in enumerate(source.splitlines(), start=1):
        m = _NON_ASCII.search(line)
        if not m:
            continue
        if _rule_24_line_exempt(path, ln):
            continue
        ch = m.group(0)
        violations.append(
            Violation(
                path,
                ln,
                m.start(),
                'rule_24',
                f"non-ASCII char {ch!r} (U+{ord(ch):04X}) in source; "
                f"use ASCII (e.g. 'degC' not the degree sign, '--' not "
                f"the em-dash, 'um' not the micro sign)",
            )
        )
    return violations


def _check_rule_24_kv(content: str, path: str) -> list[Violation]:
    """Rule 24 ASCII-only check for .kv files.

    .kv files have no AST gate; the rule covers the entire file per
    CLAUDE.md ("every string in source, every comment, every docstring
    ... in .py / .c / .h / .kv / similar code files"). Same shape as
    `_check_rule_24` for .py; separate function because .kv has no
    file-level exempt today.
    """
    violations: list[Violation] = []
    for ln, line in enumerate(content.splitlines(), start=1):
        m = _NON_ASCII.search(line)
        if not m:
            continue
        ch = m.group(0)
        violations.append(
            Violation(
                path,
                ln,
                m.start(),
                'rule_24',
                f'non-ASCII char {ch!r} (U+{ord(ch):04X}) in .kv file; '
                f"use ASCII (e.g. 'um' not the micro sign, '--' not the em-dash)",
            )
        )
    return violations


def _check_rule_28(tree: ast.Module, path: str) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not _is_notification_call(node):
            continue
        for s, ln, col in _arg_string_constants(node):
            m = _INTERNAL_ID_PATTERN.search(s)
            if not m:
                continue
            violations.append(
                Violation(
                    path,
                    ln,
                    col,
                    'rule_28',
                    f'internal ID {m.group(0)!r} in notifications string; user-facing '
                    f'strings must not include rule tags / audit IDs / fix-N refs',
                )
            )
    return violations


def _iter_comments(source: str) -> list[tokenize.TokenInfo]:
    """Return COMMENT tokens up to (but not including) the first tokenize error.

    Iterating manually rather than `list(...)`-on-the-generator preserves
    any comments emitted before a tokenize/syntax error mid-source.
    """
    comments: list[tokenize.TokenInfo] = []
    try:
        for tok in tokenize.tokenize(io.BytesIO(source.encode('utf-8')).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    return comments


def _is_test_path(path: str) -> bool:
    """Test files are exempt from Rule 27a / 27b because naming the
    bug / issue / audit finding under regression IS the point of the
    test. ``test_issue_671_...`` and docstrings that cite the failing
    code path are load-bearing for the test's purpose; Rule 27's "no
    chronology" goal is the opposite shape -- production code.

    Also exempts this rule-check file itself -- its purpose is to talk
    about the rule keywords + patterns it enforces, which inherently
    requires mentioning the words "audit", "Rule N", etc.

    Matches files under any directory named ``tests`` OR whose basename
    starts with ``test_`` (the pytest convention).
    """
    norm = path.replace('\\', '/')
    if '/tests/' in norm or norm.startswith('tests/'):
        return True
    if norm.endswith('tools/check_rules.py') or norm == 'check_rules.py':
        return True
    basename = norm.rsplit('/', 1)[-1]
    return basename.startswith('test_')


def _check_rule_27a(source: str, path: str) -> list[Violation]:
    if _is_test_path(path):
        return []
    violations: list[Violation] = []
    for tok in _iter_comments(source):
        text = tok.string
        if 'TEMP:' in text:
            continue
        m = _TODO_PATTERN.search(text)
        if not m:
            continue
        violations.append(
            Violation(
                path,
                tok.start[0],
                tok.start[1],
                'rule_27a',
                f'`# {m.group(0)}` in source; extract to docs/TODO.md or delete',
            )
        )
    return violations


def _check_rule_27b(source: str, path: str) -> list[Violation]:
    if _is_test_path(path):
        return []
    violations: list[Violation] = []
    for tok in _iter_comments(source):
        for pat, label in _COMMENT_ID_PATTERNS:
            m = pat.search(tok.string)
            if not m:
                continue
            violations.append(
                Violation(
                    path,
                    tok.start[0],
                    tok.start[1],
                    'rule_27b',
                    f'{label} {m.group(0)!r} in comment; record decisions, not chronology',
                )
            )
            break
    return violations


def _iter_docstrings(tree: ast.AST):
    """Yield (lineno, col, text) for each module / class / function
    docstring in the AST.

    Docstrings are string-literal expression statements, not COMMENT
    tokens, so the tokenize-based `_iter_comments` walk misses them.
    Rule 27 applies to docstrings too -- they end up in `help()`,
    `__doc__`, and IDE tooltips, so audit-doc IDs in docstrings are
    just as much chronology-leaking as comment refs.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            if node.body and isinstance(node.body[0], ast.Expr):
                first = node.body[0]
                yield first.lineno, first.col_offset, doc


def _check_rule_27d(tree: ast.AST, path: str) -> list[Violation]:
    """Same patterns as 27b (comments) but applied to docstrings."""
    if _is_test_path(path):
        return []
    violations: list[Violation] = []
    for lineno, col, text in _iter_docstrings(tree):
        for pat, label in _COMMENT_ID_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            violations.append(
                Violation(
                    path,
                    lineno,
                    col,
                    'rule_27d',
                    f'{label} {m.group(0)!r} in docstring; record decisions, not chronology',
                )
            )
            break
    return violations


_POST_PROCESSOR_WRITE_PATHS = frozenset({
    'modules/zprojector.py',
    'modules/stitcher.py',
    'modules/composite_generation.py',
})

_TIFFFILE_NAMES = frozenset({'tf', 'tifffile'})

_FALSE_COLOR_HELPER_NAMES = frozenset({
    'maybe_apply_false_color',
    'write_tiff',
})

_RULE_31A_PATH_SCOPE = ('modules/', 'ui/')
_RULE_31A_FILE_EXEMPT = frozenset({
    # image_utils.py owns image_file_to_image (multi-format L1 file
    # loader called from the post-processing UI + Kivy display path)
    # plus the imread_color / imwrite_color / videowriter_color
    # capability-flag wrappers. cv2 use is by definition boundary code.
    'modules/image_utils.py',
    # video_writer.py owns the cv2.VideoWriter XVID fallback for the
    # canonical VideoWriter class -- the wrapper that surrounding
    # callers consume; direct cv2.VideoWriter outside this class swaps
    # BGR / RGB at the file boundary.
    'modules/video_writer.py',
})
_RULE_31A_BANNED_CV2_ATTRS = frozenset({'imread', 'imwrite', 'VideoWriter'})

_RULE_31B_BOUNDARY_PATHS = frozenset({
    # The display / encode boundary where mono -> RGB false-color
    # widening is correct. Save / process callers must apply false
    # color via mono_to_rgb_falsecolor at the display / encode edge,
    # not at the storage edge -- mono fluorescence saves keep the
    # layer as TIFF metadata.
    'ui/main_display.py',
    'modules/video_capture.py',
})


def _check_rule_31c(tree: ast.AST, path: str) -> list[Violation]:
    """Block bare ``tf.imwrite`` / ``tifffile.imwrite`` in post-processor
    modules whose canonical save path is the false-color-aware
    ``image_utils.write_tiff`` (or its extracted helper
    ``image_utils.maybe_apply_false_color`` called before a bare
    imwrite).

    Bug shape this prevents: post-processor functions that compute a
    fluorescence-shaped output and save via bare tifffile.imwrite
    bypass the false-color RGB widening. Symptom: greyscale projection
    / stitched / composite outputs even with the false_color_16bit
    setting on.

    Per-function pairing rule: a function may call tifffile.imwrite IF
    the same function also calls one of the false-color helpers. A
    function with bare imwrite and no paired helper call fires.

    Path scope: only fires on the modules listed in
    ``_POST_PROCESSOR_WRITE_PATHS`` -- the sinks where the canonical
    route is established. Expand the set as other post-processors
    migrate.
    """
    norm = path.replace('\\', '/')
    if not any(norm.endswith(p) for p in _POST_PROCESSOR_WRITE_PATHS):
        return []
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bare_imwrite_calls: list[ast.Call] = []
        helper_seen = False
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if isinstance(f, ast.Attribute):
                if (
                    f.attr == 'imwrite'
                    and isinstance(f.value, ast.Name)
                    and f.value.id in _TIFFFILE_NAMES
                ):
                    bare_imwrite_calls.append(sub)
                elif f.attr in _FALSE_COLOR_HELPER_NAMES:
                    helper_seen = True
        if bare_imwrite_calls and not helper_seen:
            for c in bare_imwrite_calls:
                violations.append(
                    Violation(
                        path,
                        c.lineno,
                        c.col_offset,
                        'rule_31c',
                        f'bare tifffile.imwrite without a paired '
                        f'image_utils.maybe_apply_false_color (or '
                        f'image_utils.write_tiff) call in the same '
                        f'function. Post-processor outputs must apply '
                        f'the false-color gate before the bare imwrite, '
                        f'or fluorescence saves grayscale.',
                    )
                )
    return violations


def _check_rule_31a(tree: ast.AST, path: str) -> list[Violation]:
    """Block bare ``cv2.imread`` / ``cv2.imwrite`` / ``cv2.VideoWriter``
    in production ``modules/`` and ``ui/`` outside the canonical owner
    files.

    Bug shape this prevents: callers reach for ``cv2.imread`` /
    ``cv2.imwrite`` directly to read or write image files. cv2 is
    BGR-native; viewers (tifffile, FIJI, OS preview) and the rest of
    the pipeline are RGB-native, so a bare cv2 call swaps channels at
    the file boundary and silently corrupts color order. The canonical
    routes are the ``image_utils.imread_color`` /
    ``image_utils.imwrite_color`` / ``image_utils.videowriter_color``
    capability-flag wrappers (which live in ``modules/image_utils.py``)
    plus the ``modules.video_writer.VideoWriter`` wrapper class
    (which owns the cv2.VideoWriter XVID fallback).

    Path scope: only fires on ``modules/`` and ``ui/`` sources. File-
    level exempt for the canonical owner files in
    ``_RULE_31A_FILE_EXEMPT``. Test files exempt via ``_is_test_path``.
    """
    if _is_test_path(path):
        return []
    norm = path.replace('\\', '/')
    if not any(norm.startswith(scope) for scope in _RULE_31A_PATH_SCOPE):
        return []
    if norm in _RULE_31A_FILE_EXEMPT:
        return []
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        if (
            isinstance(f.value, ast.Name)
            and f.value.id == 'cv2'
            and f.attr in _RULE_31A_BANNED_CV2_ATTRS
        ):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    node.col_offset,
                    'rule_31a',
                    f'bare cv2.{f.attr} in modules/ or ui/; route through '
                    f'image_utils.imread_color / imwrite_color / '
                    f'videowriter_color (capability-flag wrappers) or the '
                    f'canonical modules.video_writer.VideoWriter class. '
                    f'cv2 is BGR-native and bare calls swap channels at '
                    f'the file boundary.',
                )
            )
    return violations


def _check_rule_31b(tree: ast.AST, path: str) -> list[Violation]:
    """Block ``add_false_color`` callsites outside the display / encode
    boundary.

    Bug shape this prevents: a save / process module widens a mono
    fluorescence frame to a 3-channel RGB replica via
    ``add_false_color`` before write. The pre-mono-native save path
    used this; the mono-native pipeline keeps the layer as TIFF
    metadata and applies false-color only at the display / encode
    boundary. Bringing ``add_false_color`` back into a save / process
    path bakes false-color into the stored file and breaks downstream
    consumers that expect mono + layer metadata.

    Path scope: any production ``.py``. Allowed call sites are listed
    in ``_RULE_31B_BOUNDARY_PATHS`` -- the manual record path and
    protocol video capture. Test files exempt via ``_is_test_path``.
    """
    if _is_test_path(path):
        return []
    norm = path.replace('\\', '/')
    if norm in _RULE_31B_BOUNDARY_PATHS:
        return []
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == 'add_false_color':
            pass
        elif isinstance(f, ast.Name) and f.id == 'add_false_color':
            pass
        else:
            continue
        boundary_list = ', '.join(sorted(_RULE_31B_BOUNDARY_PATHS))
        violations.append(
            Violation(
                path,
                node.lineno,
                node.col_offset,
                'rule_31b',
                f'add_false_color callsite outside the display / encode '
                f'boundary. Allowed call sites: {boundary_list}. Mono '
                f'fluorescence saves carry the layer as TIFF metadata; '
                f'widening to RGB at the save / process layer bakes false '
                f'color into the file. Apply false-color at the display / '
                f'encode boundary via image_utils.mono_to_rgb_falsecolor.',
            )
        )
    return violations


_BROKEN_SCOPE_METHODS = frozenset({
    # Methods that were on Lumascope pre-Wave-7 and are now ONLY reachable
    # via a sub-API namespace (scope.motion.X, scope.imaging.X, etc.).
    # Lumascope itself no longer exposes a same-named forwarder, so bare
    # `scope.<name>(...)` raises AttributeError at runtime.
    # Bench-day 2026-05-26 surfaced 68 such calls in
    # etaluma-engineering/.../camera_characterization.py that the test
    # suite missed because every test scope was a MagicMock that silently
    # absorbs any attribute access. This rule catches new occurrences at
    # pre-commit, before they ship.
    #
    # New entries get added here when a method moves off Lumascope onto a
    # sub-API. Whitelist: scope.motion.X / scope.imaging.X / scope.illumination.X /
    # scope.diagnostics.X / scope.capabilities.X / scope.io.X / scope.runtime_state.X.
    'move_absolute_position',
    'move_relative_position',
    'get_current_position',
    'get_target_position',
    'get_target_status',
    'set_motor_precision_mode',
    'set_pixel_format',
    'set_frame_size',
    'set_gain',
    'set_exposure_t',
    'set_exposure_time',
    'capture_and_wait',
    'get_channels',
    'run_pylon_diagnostic_probe',
})


def _check_rule_35d(tree: ast.AST, path: str) -> list[Violation]:
    """Block bare scope.<method> calls for methods relocated to sub-APIs.

    Wave-7 sub-API decomposition moved camera / motion / diagnostics
    methods off Lumascope onto namespaced sub-APIs (scope.imaging,
    scope.motion, etc.). Lumascope no longer exposes the bare name, so
    `scope.<name>(...)` is an AttributeError at runtime -- but tests
    using MagicMock scopes silently absorb the access. This rule
    AST-greps for the broken-name set on any Attribute access whose
    base resolves to a name 'scope' (matches both bare `scope.X` and
    `<owner>.scope.X` patterns).

    Exempts test files (intentional MagicMock targeting), the lumascope_api/
    package itself (the API surface that defines or forwards these names),
    and this rule-check file. New broken-on-Lumascope names get added to
    `_BROKEN_SCOPE_METHODS` as Wave-7 phases retire more forwarders.
    """
    norm = path.replace('\\', '/')
    if _is_test_path(norm):
        return []
    if '/lumascope_api/' in norm or norm.endswith('lumascope_api.py'):
        return []
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in _BROKEN_SCOPE_METHODS:
            continue
        base = node.value
        # Match `scope.X` (bare) and `<expr>.scope.X` (e.g. self.scope.X,
        # ctx.scope.X, lumaview.scope.X). The base of node is the thing to
        # the left of `.X`; either a Name 'scope' or an Attribute whose
        # .attr is 'scope'.
        is_scope_base = (
            (isinstance(base, ast.Name) and base.id == 'scope')
            or (isinstance(base, ast.Attribute) and base.attr == 'scope')
        )
        if not is_scope_base:
            continue
        violations.append(
            Violation(
                path,
                node.lineno,
                node.col_offset,
                'rule_35d',
                f'bare scope.{node.attr}(...) -- the method moved to a '
                f'sub-API namespace post-Wave-7. Route through '
                f'scope.motion / scope.imaging / scope.illumination / '
                f'scope.diagnostics / scope.capabilities / scope.io / '
                f'scope.runtime_state as appropriate. Lumascope no '
                f'longer exposes a same-named forwarder; MagicMock scopes '
                f'in tests will silently absorb the access but bench '
                f'hardware raises AttributeError.',
            )
        )
    return violations


def _check_rule_42(source: str, path: str) -> list[Violation]:
    """WARN on `healthy` / `fine` / `within range` in comments without a
    `PERFORMANCE_BUDGETS.md` cite anywhere in the file.

    Rule 42 says calling an observation "healthy" without citing a budget
    row is a Rule 39 violation. This catches the common case in comments;
    commit-message scanning is a future commit-msg-stage hook.
    """
    violations: list[Violation] = []
    if _RULE_42_SUPPRESS.search(source):
        return violations
    for tok in _iter_comments(source):
        m = _RULE_42_TRIGGER.search(tok.string)
        if not m:
            continue
        violations.append(
            Violation(
                path,
                tok.start[0],
                tok.start[1],
                'rule_42',
                f'{m.group(0)!r} in comment without a `PERFORMANCE_BUDGETS.md` '
                'cite; an observation called healthy needs a budget row reference',
                severity='warn',
            )
        )
    return violations


def check_source(content: str, path: str) -> list[Violation]:
    """Run all enabled rule checks against one source file's content."""
    violations: list[Violation] = []
    # Rule 24 is text-scan only -- run regardless of AST parseability.
    violations.extend(_check_rule_24(content, path))
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as e:
        violations.append(
            Violation(
                path,
                e.lineno or 1,
                e.offset or 0,
                'parse',
                f'could not parse: {e.msg}',
            )
        )
    else:
        violations.extend(_check_rule_28(tree, path))
        violations.extend(_check_rule_27d(tree, path))
        violations.extend(_check_rule_31a(tree, path))
        violations.extend(_check_rule_31b(tree, path))
        violations.extend(_check_rule_31c(tree, path))
        violations.extend(_check_rule_35d(tree, path))
    violations.extend(_check_rule_27a(content, path))
    violations.extend(_check_rule_27b(content, path))
    violations.extend(_check_rule_42(content, path))
    return violations


def _staged_python_files() -> list[str]:
    out = subprocess.check_output(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=AM'],
        text=True,
    )
    return [p for p in out.splitlines() if p.endswith('.py')]


def _staged_kv_files() -> list[str]:
    out = subprocess.check_output(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=AM'],
        text=True,
    )
    return [p for p in out.splitlines() if p.endswith('.kv')]


def _read_staged_content(path: str) -> str:
    return subprocess.check_output(['git', 'show', f':{path}'], text=True)


_HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


def _added_lines(path: str) -> set[int]:
    """Return the set of line numbers added (or modified) in path's staged diff.

    Uses --unified=0 so the diff has no context lines; every '+' in the diff
    body corresponds to a line in the new file.
    """
    out = subprocess.check_output(
        ['git', 'diff', '--cached', '--unified=0', '--', path],
        text=True,
    )
    added: set[int] = set()
    cur_new = 0
    for line in out.splitlines():
        if line.startswith('@@'):
            m = _HUNK_HEADER.match(line)
            if m:
                cur_new = int(m.group(1))
            continue
        if line.startswith('+++') or line.startswith('---'):
            continue
        if line.startswith('+'):
            added.add(cur_new)
            cur_new += 1
        elif line.startswith('-'):
            continue
        else:
            cur_new += 1
    return added


def _filter_to_added(violations: list[Violation], added: set[int]) -> list[Violation]:
    return [v for v in violations if v.line in added]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--staged', action='store_true', help='check staged content')
    src.add_argument('--paths', nargs='+', help='explicit file paths to check')
    parser.add_argument(
        '--all',
        action='store_true',
        help='flag every violation (default with --staged: only NEW lines)',
    )
    args = parser.parse_args(argv)

    violations: list[Violation] = []

    if args.staged:
        for p in _staged_python_files():
            try:
                content = _read_staged_content(p)
            except subprocess.CalledProcessError:
                continue
            file_violations = check_source(content, p)
            if not args.all:
                added = _added_lines(p)
                file_violations = _filter_to_added(file_violations, added)
            violations.extend(file_violations)
        for p in _staged_kv_files():
            try:
                content = _read_staged_content(p)
            except subprocess.CalledProcessError:
                continue
            file_violations = _check_rule_24_kv(content, p)
            if not args.all:
                added = _added_lines(p)
                file_violations = _filter_to_added(file_violations, added)
            violations.extend(file_violations)
    else:
        for p in args.paths or []:
            try:
                content = Path(p).read_text(encoding='utf-8', errors='replace')
            except OSError as e:
                print(f'{p}: cannot read: {e}', file=sys.stderr)
                continue
            if p.endswith('.kv'):
                violations.extend(_check_rule_24_kv(content, p))
            else:
                violations.extend(check_source(content, p))

    if not violations:
        return 0

    blocks = [v for v in violations if v.severity == 'block']
    warns = [v for v in violations if v.severity == 'warn']

    if blocks:
        print(f'\n{len(blocks)} rule violation(s) found:\n', file=sys.stderr)
        for v in blocks:
            print(v.format(), file=sys.stderr)

    if warns:
        print(f'\n{len(warns)} rule warning(s):\n', file=sys.stderr)
        for v in warns:
            print(v.format(), file=sys.stderr)

    print('', file=sys.stderr)
    if blocks:
        print('See docs/CLAUDE.md Rules 24, 27, 28, 42.', file=sys.stderr)
        print('To bypass (NOT recommended): git commit --no-verify', file=sys.stderr)
        return 1
    print('Warnings only; commit allowed. See docs/CLAUDE.md Rule 42.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
