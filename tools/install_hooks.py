#!/usr/bin/env python3
"""Install / uninstall the pre-commit hook in this git repo.

The hook delegates to ``tools/check_rules.py --staged`` so every commit
runs the mechanical CLAUDE.md rule checks before the commit lands.

LVP-specific: the managed hook ALSO bumps version.txt (timestamp +
branch fields) after the rule check passes, replacing the standalone
version-bump hook that lived in LVP previously. Order is intentional:
rule check first so a violation fails fast and version.txt isn't
touched on a doomed commit.

Modes:

    python tools/install_hooks.py --install
    python tools/install_hooks.py --uninstall
    python tools/install_hooks.py --dry-run   # scan whole repo, no install

Idempotent: ``--install`` against a hook this tool previously wrote
overwrites it. Against a hook this tool did NOT write (no marker), the
install refuses and asks the user to integrate manually -- repos that
already have a pre-commit hook (e.g. the standalone version.txt hook
LVP shipped before this integration) need conscious replacement, not
silent clobber.

The dry-run mode is the punch-list generator: it scans every .py file
under the repo and reports violations regardless of staged state, so a
fresh install on a repo with pre-existing violations gives a concrete
cleanup list instead of failing on the first commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HOOK_MARKER = '# managed by tools/install_hooks.py (CLAUDE.md Rule 31)'

_HOOK_SCRIPT = f"""#!/usr/bin/env bash
{_HOOK_MARKER}
# Mechanical Rule 24 / 27 / 28 pre-commit gate + version.txt bump.
# Edit tools/check_rules.py to change the checks; do NOT edit this
# hook directly (re-running tools/install_hooks.py --install will
# overwrite).
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Rule 31 mechanical gate (fail fast on Rule 24 / 27 / 28 violations
# in staged lines before any other hook side-effects). Skip gracefully on
# branches that predate the check_rules.py port (e.g. main) so the gate's
# absence never blocks a commit there.
if [ -f "$REPO_ROOT/tools/check_rules.py" ]; then
    python3 "$REPO_ROOT/tools/check_rules.py" --staged
else
    echo "pre-commit: tools/check_rules.py absent on this branch -- skipping rule gate" >&2
fi

# Ruff finding-count ratchet: blocks commits that raise the repo-wide
# ruff count above tools/ruff_baseline.txt; auto-lowers + stages the
# baseline when cleanup reduces the count. The tool itself skips
# gracefully when ruff or the baseline file is absent.
if [ -f "$REPO_ROOT/tools/ruff_ratchet.py" ]; then
    python3 "$REPO_ROOT/tools/ruff_ratchet.py" --pre-commit
fi

# version.txt refresh (LVP-specific). 4-line format:
#   Line 1: release moniker (manual bump on promotion; path-safe)
#   Line 2: commit timestamp (this hook rewrites)
#   Line 3: branch name (this hook rewrites)
#   Line 4: build GUID -- random per commit, embedded IN the commit
#           that produces it. Sidesteps the SHA chicken-and-egg: the
#           GUID does not need to match the resulting SHA; a unique
#           tag per commit is enough for log triage. Lookup via:
#               git log -S "<guid>" -- version.txt
# Lines 2+3 give triage "branch + timestamp" identity for bench bundles;
# Line 4 gives an exact commit lookup that works in any distribution
# (ZIP, clone, installer alike) without depending on GitHub or git
# archive substitution.
VERSION_FILE="$REPO_ROOT/version.txt"
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(head -1 "$VERSION_FILE")
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GUID=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])" 2>/dev/null \\
        || openssl rand -hex 4 2>/dev/null \\
        || echo "nogenuid")
    printf "%s\\n%s\\n%s\\n%s\\n" "$VERSION" "$TIMESTAMP" "$BRANCH" "$GUID" > "$VERSION_FILE"
    git add "$VERSION_FILE"
fi
"""

# Directories whose .py files are not subject to the rule check.
_EXCLUDE_DIR_NAMES = frozenset(
    {
        '.git',
        '__pycache__',
        'build',
        'dist',
        'venv',
        '.venv',
        'env',
        '.env',
        'node_modules',
        'completed',
    }
)


def _git_dir() -> Path:
    """Return the resolved .git directory for the current repo."""
    out = subprocess.check_output(['git', 'rev-parse', '--git-dir'], text=True).strip()
    return Path(out).resolve()


def _repo_root() -> Path:
    out = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
    return Path(out).resolve()


def _hook_path() -> Path:
    return _git_dir() / 'hooks' / 'pre-commit'


def install() -> int:
    hook = _hook_path()
    if hook.exists():
        existing = hook.read_text(encoding='utf-8', errors='replace')
        if _HOOK_MARKER not in existing:
            print(
                f'ERROR: {hook} already exists and was not installed by '
                f'this tool.\n'
                f'  Refusing to overwrite. Integrate the rule-check call '
                f'into the existing hook manually:\n'
                f'    python3 "$(git rev-parse --show-toplevel)"'
                f'/tools/check_rules.py --staged\n'
                f'  Or remove the existing hook and re-run --install.',
                file=sys.stderr,
            )
            return 2
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(_HOOK_SCRIPT, encoding='utf-8')
    hook.chmod(0o755)
    print(f'Installed pre-commit hook at {hook}')
    print('  Hook delegates to tools/check_rules.py --staged.')
    print('  To bypass for one commit: git commit --no-verify')
    print('  To remove: tools/install_hooks.py --uninstall')
    return 0


def uninstall() -> int:
    hook = _hook_path()
    if not hook.exists():
        print(f'No pre-commit hook at {hook}; nothing to uninstall.')
        return 0
    existing = hook.read_text(encoding='utf-8', errors='replace')
    if _HOOK_MARKER not in existing:
        print(
            f'ERROR: {hook} exists but was not installed by this tool.\n'
            f'  Refusing to remove. Inspect manually.',
            file=sys.stderr,
        )
        return 2
    hook.unlink()
    print(f'Removed pre-commit hook at {hook}')
    return 0


def _walk_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _EXCLUDE_DIR_NAMES:
                    continue
                if entry.name.startswith('.') and entry.name != '.':
                    continue
                stack.append(entry)
            elif entry.is_file() and entry.suffix == '.py':
                out.append(entry)
    out.sort()
    return out


def dry_run() -> int:
    """Scan every .py file in the repo and report violations.

    Uses tools/check_rules.py --paths so the same rule logic runs for
    both modes. Prints the file:line:rule list to stderr; returns 1
    if any violations found, 0 otherwise.
    """
    root = _repo_root()
    files = _walk_python_files(root)
    if not files:
        print('No .py files found.')
        return 0
    rel_paths = [str(f.relative_to(root)) for f in files]
    checker = root / 'tools' / 'check_rules.py'
    if not checker.exists():
        print(f'ERROR: {checker} not found.', file=sys.stderr)
        return 2
    print(f'Scanning {len(rel_paths)} Python file(s) under {root}')
    rc = subprocess.run(
        [sys.executable, str(checker), '--paths', *rel_paths],
        cwd=root,
    ).returncode
    if rc == 0:
        print('No Rule 24 / 27 / 28 violations found.')
    else:
        print(
            '\n(Above violations are the cleanup punch list. The hook '
            'only blocks NEW violations on staged lines by default; '
            'pre-existing violations like these are surfaced here for '
            'visibility but do not block commits.)'
        )
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--install', action='store_true', help='install pre-commit hook')
    mode.add_argument(
        '--uninstall', action='store_true', help='remove the hook installed by this tool'
    )
    mode.add_argument(
        '--dry-run', action='store_true', help='scan whole repo for violations; do not install'
    )
    args = parser.parse_args(argv)

    if args.install:
        return install()
    if args.uninstall:
        return uninstall()
    if args.dry_run:
        return dry_run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
