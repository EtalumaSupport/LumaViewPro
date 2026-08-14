"""No file may still declare a superseded Python floor.

WHY THIS IS A REPO-WIDE SEARCH AND NOT A LIST OF KNOWN SITES

On 2026-06-02 the dependency floor moved to 3.12 (imagecodecs 2026.5.10
dropped 3.11). The commit that did it was a deliberate lockstep sweep working
from a requirements audit, and it correctly updated the ruff target, the
Windows helper's SUPPORTED tuple, README and three install scripts. Eight
declarations were still saying 3.11 seventy-three days later, and the CI job
added in that window could not install a single dependency as a result.

It failed in two ways that more care would not have prevented:

  1. Its file list omitted the application entry point and the L2 integration
     doc, so nothing inside those files could be found by any amount of
     diligence inside the sweep.
  2. Inside files it DID edit, distance from the edit decided survival.
     install_mac.sh lost `for minor in 13 12 11` and kept `-ge 11` twenty
     lines below it; install_linux.sh had both error messages corrected while
     MIN_MINOR=11 survived.

A guard that enumerates known declaration sites is the same artifact as the
audit list that already failed -- it cannot catch cause 1 by construction.
Only a search over every tracked file can. Hence `git grep`.

WHAT THIS IS, HONESTLY

A superseded-value detector, not a single source of truth. It cannot make the
illegal state unrepresentable; it makes one specific outdated value loud.
Storing the floor in one place was considered and rejected twice: a root
`.python-version` is read by pyenv/uv/rye/mise and would silently retarget
developers' interpreters, and an importable Python constant cannot be read by
the shell installers, whose whole job is to run before a usable interpreter
exists.

WHEN THE FLOOR MOVES AGAIN

Set _SUPERSEDED_MINOR to the floor you are leaving. The test then names every
site still on it -- which is precisely the completeness check the 2026-06
sweep did not have. Verify the false-positive load first: a bare `3.N` matches
gains, sizes and third-party versions, which is why the patterns below are
declaration idioms rather than a bare version string.
"""

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The floor we left behind. Not the current floor -- this test asserts the
# absence of the OLD value, so it needs no authoritative store of the new one.
_SUPERSEDED_MINOR = 11

# Declaration idioms only. A bare `3.11` happens to have no false positives
# today, but the same pattern shape at `3.10` matches "Stage 3.6", "3.8 GB"
# and matplotlib 3.10.9, so the set stays anchored to how a floor is actually
# written down rather than to a bare version string.
_PATTERNS = [
    rf'3\.{_SUPERSEDED_MINOR}',  # prose, YAML, requires-python, docs
    rf'3, {_SUPERSEDED_MINOR}',  # sys.version_info tuple comparison
    rf'py3{_SUPERSEDED_MINOR}',  # ruff target-version
    rf'MIN_MINOR={_SUPERSEDED_MINOR}',  # install_linux.sh
    rf'-ge {_SUPERSEDED_MINOR} ',  # install_mac.sh numeric bound
    rf'-lt {_SUPERSEDED_MINOR} ',
]

# Files permitted to mention the superseded floor, each with the reason it is
# legitimate. Empty on purpose: the census at the time of writing was 9 hits
# before the fix and 0 after, with no false positives, so there is nothing to
# excuse yet. Release notes recording "dropped 3.11" would be the first real
# entry -- add it here with its reason rather than widening the patterns.
_ALLOWED: set[str] = set()


def _git_grep(pattern):
    """Tracked-file search. Returns [] on no match (git grep exits 1)."""
    # -e is required, not stylistic: patterns like `-ge 11 ` start with a dash
    # and git parses them as switches otherwise, so the guard dies instead of
    # searching.
    result = subprocess.run(
        ['git', 'grep', '-nE', '-e', pattern],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f'git grep failed for {pattern!r}: {result.stderr.strip()}')
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_no_file_declares_a_superseded_python_floor():
    offenders = []
    for pattern in _PATTERNS:
        for hit in _git_grep(pattern):
            path = hit.split(':', 1)[0]
            if path in _ALLOWED or path == f'tests/{Path(__file__).name}':
                continue
            offenders.append(hit)

    assert not offenders, (
        f'{len(offenders)} site(s) still declare Python 3.{_SUPERSEDED_MINOR} as a '
        f'supported floor:\n  ' + '\n  '.join(sorted(set(offenders))) + '\n\n'
        'Every declaration must move together -- the installers, the startup '
        'gate, the workflow and the docs each turn a stale floor into a '
        'different broken experience. If one of these is a legitimate '
        'historical reference (release notes, a changelog entry), add its path '
        f'to _ALLOWED in tests/{Path(__file__).name} with the reason.'
    )
