"""No module the run engine imports reads either process-wide global.

The run engine and the collaborators it composes used to take thirteen
values from two process globals -- the GUI's application context and the
settings module's import-time binding -- instead of from the session and the
run plan. In the GUI both globals alias the live state, so nothing showed; in
any headless process both are None, and the same run crashed, wrote a
different root, dropped an exposure ceiling, lost filename tokens, omitted
the image scale, swallowed the autofocus restore and resolved a different
output encoding.

Every one of those reads is now gone, and this guard keeps them gone. It walks
the engine's import graph from the sequenced-capture runner and asserts the
two reaches it still contains against an allowlist derived from the record:
the two telemetry modules that read the context (capture fps off the display
widget, executor queue sizes for the memory metrics -- guarded, empty when
unset, and headless already gets the right answer), and the one startup
composition read of the settings binding (the simulator's default model). A
third name in either list is a regression.

What this guard claims, exactly: no module the run engine IMPORTS reads either
global. It cannot see a read in a module outside the walk, and is not meant
to -- the GUI post-processing resolver for the live image mode lives in such
a module by design.
"""

import ast
import pathlib
import re

from tests.ast_seams import REPO_ROOT

ENGINE_ROOT_MODULE = 'modules.sequenced_capture_runner'

# The reach of the application context the engine set may keep: telemetry,
# guarded, correct headless.
ALLOWED_CONTEXT_READERS = frozenset({'modules.config_helpers', 'modules.metrics_logger'})
# The one import-time binding of the settings module the engine set may keep:
# the simulated motor board's default model, read once at scope construction.
ALLOWED_SETTINGS_BINDERS = frozenset({'modules.lumascope_api._lumascope'})

_CONTEXT_IMPORT = re.compile(
    r'import\s+modules\.app_context|from\s+modules\s+import\s+app_context|'
    r'from\s+modules\.app_context\s+import'
)
# A BIND of the global -- not a function that re-reads the settings FILE,
# which imports a different name from the same module.
_SETTINGS_BIND = re.compile(r'from\s+modules\.settings_init\s+import\s+settings\b')


def _module_path(name: str) -> pathlib.Path:
    return REPO_ROOT / (name.replace('.', '/') + '.py')


def _imports_of(path: pathlib.Path) -> set[str]:
    """Every module name an import statement in the file names, at any scope."""
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f'{node.module}.{alias.name}' for alias in node.names)
    return out


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
            for dep in _imports_of(_module_path(name))
            if dep.startswith('modules.') and dep not in seen
        )
    return seen


def _matching(pattern: re.Pattern, modules: set[str]) -> set[str]:
    return {name for name in modules if pattern.search(_module_path(name).read_text())}


def test_the_walk_reaches_the_engine():
    """The guard is only as good as its walk: the engine's own files must be
    in the set it checks, or an empty list proves nothing."""
    modules = engine_set()
    for name in (
        'modules.sequenced_capture_runner',
        'modules.protocol_step_runner',
        'modules.protocol_image_writer',
        'modules.protocol_cleanup',
        'modules.image_save',
        'modules.common_utils',
        'modules.image_utils',
        'modules.path_utils',
        'modules.composite_generation',
    ):
        assert name in modules, f'{name} is not in the engine set the guard walks'


def test_only_the_allowed_modules_reach_the_application_context():
    readers = _matching(_CONTEXT_IMPORT, engine_set())
    assert readers == set(ALLOWED_CONTEXT_READERS), (
        f'engine-set modules reaching modules.app_context: {sorted(readers)}; '
        f'allowed: {sorted(ALLOWED_CONTEXT_READERS)}'
    )


def test_only_the_allowed_module_binds_the_settings_global():
    binders = _matching(_SETTINGS_BIND, engine_set())
    assert binders == set(ALLOWED_SETTINGS_BINDERS), (
        f'engine-set modules binding settings_init.settings: {sorted(binders)}; '
        f'allowed: {sorted(ALLOWED_SETTINGS_BINDERS)}'
    )


def test_the_gui_encoding_resolver_stays_outside_the_engine():
    """The one context-reading module this work created must never be
    imported by the engine: it exists so the engine does not have to."""
    assert 'modules.derived_output_encoding' not in engine_set()
