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
    rule_24  -- ASCII-only in strings passed to logger / print / notifications
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


def _is_logger_or_notification_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id in _PRINT_NAMES:
        return True
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        base = fn.value.id
        attr = fn.attr
        if base in _LOGGER_BASES and attr in _LOGGER_METHODS:
            return True
        if base in _NOTIFICATIONS_BASES and attr in _NOTIFICATIONS_METHODS:
            return True
    return False


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


def _check_rule_24(tree: ast.Module, path: str) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not _is_logger_or_notification_call(node):
            continue
        for s, ln, col in _arg_string_constants(node):
            m = _NON_ASCII.search(s)
            if not m:
                continue
            ch = m.group(0)
            violations.append(
                Violation(
                    path,
                    ln,
                    col,
                    'rule_24',
                    f'non-ASCII char {ch!r} (U+{ord(ch):04X}) in logger/print/notification '
                    f"string; use ASCII (e.g. 'degC' not the degree sign)",
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
        violations.extend(_check_rule_24(tree, path))
        violations.extend(_check_rule_28(tree, path))
        violations.extend(_check_rule_27d(tree, path))
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
    else:
        for p in args.paths or []:
            try:
                content = Path(p).read_text(encoding='utf-8', errors='replace')
            except OSError as e:
                print(f'{p}: cannot read: {e}', file=sys.stderr)
                continue
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
