"""Every logger LumaViewPro names is ``LVP`` or a name under it.

LumaViewPro logs everything under one logger, ``LVP``, the main log's own: the
main log's floor (``lvp_logger._MainLogFloor``) passes a record below INFO only
from under it, the session puts its level on it
(``log_helpers.apply_session_level``), and in an integrator's process the log
files hang on it. A logger named anywhere else, a second root as
``lvp_logger`` was or a module's ``__name__``, loses its records below INFO
with no error, gets no session level, and is one more name an integrator must
know to capture LumaViewPro's records.

The walk reads every ``getLogger`` call in LumaViewPro's code: ``src``, ``ui``,
``tools`` and the root's ``.py`` files. A name is a string literal or a
module-level string constant of the same file; root (no argument) and Kivy's
own ``kivy`` are not LumaViewPro's names and pass by name; any other argument
is refused, since the walk cannot judge it.
"""

import ast

from tests.ast_seams import REPO_ROOT, Words, tracked_files

# Names that are not LumaViewPro's to choose: the framework's own logger.
_NOT_OURS = Words({'kivy'})


def _sources():
    return tracked_files(['src', 'ui', 'tools']) + tracked_files(pattern='*.py')


def _module_constants(tree):
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _calls(tree):
    """Each getLogger call's line and name: '' for root, None when the walk cannot read it."""
    constants = _module_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
        if called != 'getLogger':
            continue
        if not node.args and not node.keywords:
            yield node.lineno, ''
            continue
        argument = node.args[0] if node.args else node.keywords[0].value
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            yield node.lineno, argument.value
        elif isinstance(argument, ast.Name) and argument.id in constants:
            yield node.lineno, constants[argument.id]
        else:
            yield node.lineno, None


def _is_allowed(name):
    return name == '' or name in _NOT_OURS or name == 'LVP' or name.startswith('LVP.')


def test_every_logger_lumaviewpro_names_is_under_lvp():
    sources = _sources()
    named = [
        (path, line, name)
        for path in sources
        for line, name in _calls(ast.parse((REPO_ROOT / path).read_text(encoding='utf-8')))
    ]
    assert any(name == 'LVP.serial' for _, _, name in named), (
        'the walk finds no area logger: it reads too little to judge'
    )
    offenders = [
        f'{path}:{line} ' + (repr(name) if name is not None else 'a name the walk cannot read')
        for path, line, name in named
        if name is None or not _is_allowed(name)
    ]
    assert not offenders, (
        "a logger outside LumaViewPro's one root 'LVP': its records below INFO "
        'never reach the main log and it takes no session level\n' + '\n'.join(offenders)
    )
