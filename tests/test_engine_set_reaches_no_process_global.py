"""No module the run engine imports reads the application context.

The run engine and the collaborators it composes used to take thirteen
values from two process globals -- the GUI's application context and the
settings module's import-time binding -- instead of from the session and the
run plan. In the GUI both globals alias the live state, so nothing showed; in
any headless process both are None, and the same run crashed, wrote a
different root, dropped an exposure ceiling, lost filename tokens, omitted
the image scale, swallowed the autofocus restore and resolved a different
output encoding.

Every one of those reads is now gone. The settings module's binding no
longer exists, so no import of it can be written; the context remains, and
this guard keeps the engine off it. It walks the engine's import graph from
the sequenced-capture runner and asserts its reaches of the context against
an allowlist derived from the record (empty: the metrics telemetry, its
last reader, is handed what it logs). A name in the list is a regression.

What this guard claims, exactly: no module the run engine IMPORTS reads the
context. It cannot see a read in a module outside the walk, and is not meant
to -- the GUI post-processing resolver for the live image mode lives in such
a module by design.
"""

import ast
import pathlib

from tests.ast_seams import REPO_ROOT, parse_module

ENGINE_ROOT_MODULE = '_lumascope.modules.sequenced_capture_runner'

# The engine set reaches the application context nowhere: the metrics
# telemetry, its last reader, is handed what it logs.
ALLOWED_CONTEXT_READERS: frozenset[str] = frozenset()


def _rel_path(name: str) -> str:
    """The file of a module in the API's package, which sits in src/."""
    return 'src/' + name.replace('.', '/') + '.py'


def _module_path(name: str) -> pathlib.Path:
    return REPO_ROOT / _rel_path(name)


def _import_nodes(name: str):
    """Every import statement in the module, at any scope."""
    for node in ast.walk(parse_module(_rel_path(name))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def _imports_of(name: str) -> set[str]:
    """Every module name an import statement in the module names."""
    out = set()
    for node in _import_nodes(name):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif node.module and node.level == 0:
            out.add(node.module)
            out.update(f'{node.module}.{alias.name}' for alias in node.names)
    return out


def _reaches_context(name: str) -> bool:
    """``import _lumascope.modules.app_context``, ``from _lumascope.modules import app_context`` or
    ``from _lumascope.modules.app_context import ...``, at any scope."""
    for node in _import_nodes(name):
        if isinstance(node, ast.Import):
            if any(alias.name == '_lumascope.modules.app_context' for alias in node.names):
                return True
        elif node.module == '_lumascope.modules.app_context' or (
            node.module == '_lumascope.modules' and any(a.name == 'app_context' for a in node.names)
        ):
            return True
    return False


def engine_set() -> set[str]:
    """The modules reachable from the engine root through ``modules.*`` imports
    that resolve to a file."""
    seen: set[str] = set()
    stack = [ENGINE_ROOT_MODULE]
    while stack:
        name = stack.pop()
        if name in seen or not _module_path(name).exists():
            continue
        seen.add(name)
        stack.extend(
            dep
            for dep in _imports_of(name)
            if dep.startswith('_lumascope.modules.') and dep not in seen
        )
    return seen


def test_the_walk_reaches_the_engine():
    """The guard is only as good as its walk: the engine's own files must be
    in the set it checks, or an empty list proves nothing."""
    modules = engine_set()
    for name in (
        '_lumascope.modules.sequenced_capture_runner',
        '_lumascope.modules.protocol_step_runner',
        '_lumascope.modules.protocol_image_writer',
        '_lumascope.modules.protocol_cleanup',
        '_lumascope.modules.image_save',
        '_lumascope.modules.common_utils',
        '_lumascope.modules.image_utils',
        '_lumascope.modules.path_utils',
    ):
        assert name in modules, f'{name} is not in the engine set the guard walks'


def test_only_the_allowed_modules_reach_the_application_context():
    readers = {name for name in engine_set() if _reaches_context(name)}
    assert readers == set(ALLOWED_CONTEXT_READERS), (
        f'engine-set modules reaching _lumascope.modules.app_context: {sorted(readers)}; '
        f'allowed: {sorted(ALLOWED_CONTEXT_READERS)}'
    )
