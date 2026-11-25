# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No message a person reads tells them to read the log.

The log is ours, not the person's. Thirty-one messages -- popups, a run's
ending, a REST problem's detail, a status label -- once ended "check the
log" where the code had dropped the cause it could have stated. Each now
says the cause, or, for LumaViewPro's own fault, asks for a support report,
which carries the log to us. This guard reads every string in production
that is neither a docstring nor a logger call's argument, so a message
written with the advice fails here, by file and line, before a person
reads it.

A sentence is what advises; a string with no whitespace is a name or a
path (the log file's own name in ``lvp_logger``), never read for advice.
"""

import ast

from tests import ast_seams
from tests.ast_seams import MemberNames, Words

_ADVICE = (
    'check the log',
    'see the log',
    'and the log',
    'in the log',
    'the log has',
    'the log names',
    "step's log",
    'lumaviewpro.log',
    '(see log)',
    'main log',
)

_LOG_METHODS = Words({'debug', 'info', 'warning', 'warn', 'error', 'exception', 'critical', 'log'})

# The one exclusion: a failed Windows driver repair names the driver
# step's own log, which the support report may not carry; whether it does
# is read on a Windows bench, and its words change with that read.
_EXCLUDED_CLASSES = MemberNames(
    'src/_lumascope/modules/exceptions.py', {'CameraDriverRepairFailedError'}
)


def _is_logger_call(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOG_METHODS
    ):
        return False
    receiver = node.func.value
    name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, 'id', '')
    name = name.lower()
    return name.endswith('logger') or name in ('log', '_log')


def _docstring(node: ast.AST) -> ast.AST | None:
    body = getattr(node, 'body', None)
    if (
        isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value
    return None


def _advice_in(tree: ast.Module) -> list[tuple[int, str, str]]:
    """``(line, enclosing class, phrasing)`` for each advising string in ``tree``."""
    unread = set()
    for node in ast.walk(tree):
        doc = _docstring(node)
        if doc is not None:
            unread.add(id(doc))
        if _is_logger_call(node):
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                unread.update(id(n) for n in ast.walk(arg))

    found = []

    def visit(node: ast.AST, cls: str) -> None:
        if isinstance(node, ast.ClassDef):
            cls = node.name
        if id(node) not in unread:
            text = None
            if isinstance(node, ast.JoinedStr):
                text = ''.join(v.value for v in node.values if isinstance(v, ast.Constant))
                unread.update(id(v) for v in node.values)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            if text is not None and any(c.isspace() for c in text.strip()):
                said = ' '.join(text.lower().split())
                for phrasing in _ADVICE:
                    if phrasing in said:
                        found.append((node.lineno, cls, phrasing))
                        break
        for child in ast.iter_child_nodes(node):
            visit(child, cls)

    visit(tree, '')
    return found


def test_no_message_a_person_reads_sends_them_to_the_log():
    modules = list(ast_seams.iter_package_modules(('src', 'ui')))
    modules.append(('lumaviewpro.py', ast_seams.parse_module('lumaviewpro.py')))

    advising, excluded = [], set()
    for rel, tree in modules:
        for line, cls, phrasing in _advice_in(tree):
            if cls in _EXCLUDED_CLASSES:
                excluded.add(cls)
            else:
                advising.append(f'{rel}:{line} ({phrasing!r})')

    # The instrument on a known positive: the excluded class's message
    # still advises, and is found; once its words change, the exclusion
    # goes with it.
    assert excluded == _EXCLUDED_CLASSES, (
        'these exclusions no longer advise; remove them from _EXCLUDED_CLASSES: '
        + ', '.join(sorted(_EXCLUDED_CLASSES - excluded))
    )
    assert not advising, (
        'these messages send a person to the log, which is ours: state the cause, '
        "or for LumaViewPro's own fault the support-report sentence "
        '(INTERNAL_FAULT_SENTENCE):\n  ' + '\n  '.join(advising)
    )
