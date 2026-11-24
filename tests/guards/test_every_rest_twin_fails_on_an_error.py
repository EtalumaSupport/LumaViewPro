# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every REST twin under ``docs/api_examples/`` ends on an answer it did not expect.

A twin a reader runs is a shell script: without ``set -euo pipefail`` it runs
on past a failed line, and a ``curl`` without ``--fail-with-body`` exits 0 on
a 4xx or 5xx, so the next line takes a problem body's ``null`` for an id and
the script ends green having done nothing it says. So each twin sets
``set -euo pipefail`` before its first ``curl``, and every ``curl`` it calls
carries ``--fail-with-body``, which still prints the problem it was answered.
A refusal a twin shows on purpose is handled at its own line; that it is
the refusal named is the twin's own check, run by ``tests/test_api_examples.py``.
"""

from __future__ import annotations

import shlex

from tests.ast_seams import REPO_ROOT, tracked_files

_STRICT = 'set -euo pipefail'
_FAIL = '--fail-with-body'
# The shell's operators, as shlex splits them out: each one ends a command.
_OPERATOR_CHARS = set('();<>|&')


def _commands(line):
    """The words of each simple command on ``line``, its comment left out."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    command = []
    for token in lexer:
        if set(token) <= _OPERATOR_CHARS:
            if command:
                yield command
            command = []
        else:
            command.append(token)
    if command:
        yield command


def _curl_calls(line):
    """The words of each ``curl`` call on ``line``, from ``curl`` to the command's end."""
    for command in _commands(line):
        for i, word in enumerate(command):
            # ``NAME=$(curl ...`` splits as ``NAME=$`` ``(`` ``curl``.
            if word == 'curl':
                yield command[i:]
                break


def twin_failures(label, text):
    """A line per way ``text``, a twin, can run on past a failure."""
    failures = []
    lines = text.splitlines()
    first_curl = next((n for n, line in enumerate(lines, 1) if any(_curl_calls(line))), None)
    strict = next((n for n, line in enumerate(lines, 1) if line.strip() == _STRICT), None)
    if first_curl is None:
        failures.append(f'  {label}  calls no curl')
    elif strict is None or strict > first_curl:
        failures.append(f'  {label}  has no `{_STRICT}` before its first curl, line {first_curl}')
    for lineno, line in enumerate(lines, 1):
        for call in _curl_calls(line):
            if _FAIL not in call:
                failures.append(f'  {label}:{lineno}  a curl without {_FAIL}')
    return failures


def test_every_twin_sets_strict_mode_and_every_curl_fails_with_its_body():
    twins = tracked_files(('docs/api_examples',), '*.sh')
    assert twins, 'no REST twin found under docs/api_examples'
    failures = [
        failure
        for rel in twins
        for failure in twin_failures(rel, (REPO_ROOT / rel).read_text(encoding='utf-8'))
    ]
    assert not failures, (
        'A REST twin can run on past an answer it did not expect. Put '
        f'`{_STRICT}` before its first curl and {_FAIL} on every curl; a refusal '
        'shown on purpose is kept and checked at its own line.\n' + '\n'.join(failures)
    )


def test_a_twin_that_runs_on_past_a_failure_is_reported():
    planted = (
        '#!/usr/bin/env bash\n'
        '# curl in a comment is no call\n'
        'ID=$(curl -s --json "{}" $API/a | jq -r .id)   # 202\n'
        'set -euo pipefail\n'
        'curl --fail-with-body -sS $API/jobs/$ID | jq -c . && curl -sS $API/b   # 200\n'
    )
    assert twin_failures('planted.sh', planted) == [
        '  planted.sh  has no `set -euo pipefail` before its first curl, line 3',
        '  planted.sh:3  a curl without --fail-with-body',
        '  planted.sh:5  a curl without --fail-with-body',
    ]
    assert twin_failures('quiet.sh', 'set -euo pipefail\necho no call\n') == [
        '  quiet.sh  calls no curl'
    ]
