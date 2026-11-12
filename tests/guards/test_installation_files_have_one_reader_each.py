# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each file the installation ships is named in production code only by its owner.

scopes.json was read by four modules and the GUI, and the motor defaults by
three sites, each from the installation's default folder, so a scope
started on another folder read its catalogues from one folder and its
models and motor defaults from another. The scope now reads them once, in
``Lumascope._read_catalogues``, and everything else reads its copy. A file
name in any other production call is a second reader; this walk refuses it
at commit.

Two owners stay outside the scope by design: the release layer vocabulary
(``channel_record.release_channels``, and beside it the old names it replaced,
``channel_record.release_old_names``), which the settings bring-up needs
before any scope exists, and the firmware bench tool, which drives a motor
board with no scope at all.

Every installation file is parsed by one function, ``read_installation_file``,
which refuses what no installation file can mean (a key twice, a number that
is not finite). The camera timing records were parsed by a second
``json.load`` in the driver, so they were spared those refusals. Each
production ``json.load`` / ``json.loads`` is listed below with what it parses;
a new one fails here until it is listed, and a file the installation ships
is never listed: it goes through the reader.
"""

from __future__ import annotations

import ast

from tests.ast_seams import (
    iter_package_modules,
    parse_module,
    production_modules,
    tracked_files,
    walk_defs,
)

# (module, qualname) -> what its json.load / json.loads parses. None of them
# is a file the installation ships.
_JSON_PARSERS = {
    ('src/_lumascope/modules/path_utils.py', 'read_installation_file'): 'the installation files',
    ('src/_lumascope/modules/settings_init.py', 'read_settings_json'): "the user's settings",
    ('src/_lumascope/modules/post_processing.py', 'load_cell_count_method'): (
        "a user's saved cell-count method"
    ),
    ('src/_lumascope/modules/protocol.py', '_dict_cell'): "a protocol file's cell",
    ('src/_lumascope/modules/image_utils.py', '_read_ome_input_metadata'): (
        "a saved image's own metadata"
    ),
    ('src/_lumascope/modules/video_builder.py', 'VideoBuilder._read_recording_manifest'): (
        "a recording's manifest"
    ),
    ('src/_lumascope/modules/tech_support_report.py', 'validate_motorconfig'): (
        "the board's reported config"
    ),
    ('src/_lumascope/drivers/motorboard.py', 'MotorBoard.get_config'): (
        "the board's reported config"
    ),
    ('lumaviewpro.py', '<module>'): "a unit's motorconfig.json, for the simulated scope",
    ('src/_lumascope/drivers/sim_wire/backend.py', 'MotorBoardSpec.motorconfig'): (
        "the simulated board's own config"
    ),
    ('src/_lumascope/drivers/sim_wire/mp/channel.py', 'parse_state'): (
        "the simulated board's state"
    ),
    ('src/_lumascope/drivers/sim_wire/mp/machine.py', '_the_board'): "the simulated board's file",
    ('src/_lumascope/drivers/sim_wire/mp/tmc5072.py', 'build'): "the simulated board's config",
    ('src/_lumascope/rest/events.py', 'EventStream.held'): "an event's own encoded data",
    ('src/_lumascope/rest/routes.py', '_field'): "a default's own encoded form",
    ('ui/walk_file.py', 'parse_walk'): 'a walk script',
    ('tools/profiling/compare.py', 'main'): 'two profile files',
    ('tools/profiling/ground_truth.py', '_profile_workload'): "a profiler's output",
    ('tools/profiling/profile_session.py', '_config_snapshot'): "a profiled host's config",
}

_FILES = ('scopes.json', 'motorconfig_defaults.json', 'labware.json', 'objectives.json')

# (module, qualname) -> the files that def may name.
_OWNERS = {
    ('src/_lumascope/modules/lumascope_api/_lumascope.py', 'Lumascope._read_catalogues'): {
        'scopes.json',
        'motorconfig_defaults.json',
    },
    ('src/_lumascope/modules/labware_loader.py', 'WellPlateLoader.__init__'): {'labware.json'},
    ('src/_lumascope/modules/objectives_loader.py', 'ObjectiveLoader.__init__'): {
        'objectives.json'
    },
    ('src/_lumascope/modules/channel_record.py', 'release_channels'): {'scopes.json'},
    ('src/_lumascope/modules/channel_record.py', 'release_old_names'): {'scopes.json'},
    ('src/_lumascope/modules/channel_record.py', 'load_scope_models'): {'scopes.json'},
    ('tools/firmware_tools.py', '_connect_motor_board'): {'motorconfig_defaults.json'},
}


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('src/_lumascope/drivers', 'tools'))


def _named_in_calls(tree: ast.Module):
    """Yield (qualname or '<module>', lineno, file) for every file name passed to a call."""
    seen = set()

    def names(node):
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for arg in [*call.args, *(k.value for k in call.keywords)]:
                if isinstance(arg, ast.Constant) and arg.value in _FILES and id(arg) not in seen:
                    seen.add(id(arg))
                    yield arg.lineno, arg.value

    for qualname, node in walk_defs(tree.body):
        for lineno, name in names(node):
            yield qualname, lineno, name
    for lineno, name in names(tree):
        yield '<module>', lineno, name


def _load_scope_models_calls(tree: ast.Module):
    for qualname, node in walk_defs(tree.body):
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                func = call.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
                if name == 'load_scope_models':
                    yield qualname, call.lineno


def test_only_the_owners_name_an_installation_file():
    found = []
    for rel_path, tree in _all_production_modules():
        for qualname, lineno, name in _named_in_calls(tree):
            if name in _OWNERS.get((rel_path, qualname), set()):
                continue
            found.append(f'{rel_path}:{lineno} {qualname} names {name}')
    assert found == [], (
        "a second reader of an installation file; read the scope's copy "
        f'(scope.scope_models, scope.wellplate_loader, scope.objective_helper) instead: {found}'
    )


def test_only_the_scope_reads_the_model_catalogue():
    found = [
        f'{rel_path}:{lineno} {qualname}'
        for rel_path, tree in _all_production_modules()
        for qualname, lineno in _load_scope_models_calls(tree)
        if (rel_path, qualname)
        != ('src/_lumascope/modules/lumascope_api/_lumascope.py', 'Lumascope._read_catalogues')
    ]
    assert found == [], f'read scope.scope_models instead of the file: {found}'


def test_the_guard_sees_every_owner_it_exempts():
    # Each exemption must name a real def that names its files, or a rename
    # would leave the guard exempting nothing while its main check passed.
    trees = dict(_all_production_modules())
    for (rel_path, qualname), files in _OWNERS.items():
        named = {name for q, _lineno, name in _named_in_calls(trees[rel_path]) if q == qualname}
        assert named == files, f'{rel_path} {qualname} names {named}, expected {files}'


def _every_python_module():
    yield from iter_package_modules(('src', 'ui', 'tools'))
    for rel_path in tracked_files(pattern='*.py'):
        yield rel_path, parse_module(rel_path)


def _json_parses(tree: ast.Module):
    """Yield (qualname or '<module>', lineno) for every json.load / json.loads call, and every
    ``from json import`` (a bare ``load`` this walk could not tell from another)."""
    aliases = {'json'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(a.asname for a in node.names if a.name == 'json' and a.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == 'json':
            yield '<module>', node.lineno

    def parses(node):
        for call in ast.walk(node):
            func = call.func if isinstance(call, ast.Call) else None
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ('load', 'loads')
                and isinstance(func.value, ast.Name)
                and func.value.id in aliases
            ):
                yield call

    in_a_def = set()
    for qualname, node in walk_defs(tree.body):
        for call in parses(node):
            if id(call) not in in_a_def:
                in_a_def.add(id(call))
                yield qualname, call.lineno
    for call in parses(tree):
        if id(call) not in in_a_def:
            yield '<module>', call.lineno


def test_every_json_parse_is_listed_with_what_it_parses():
    found = sorted(
        f'{rel_path}:{lineno} {qualname}'
        for rel_path, tree in _every_python_module()
        for qualname, lineno in _json_parses(tree)
        if (rel_path, qualname) not in _JSON_PARSERS
    )
    assert found == [], (
        'a JSON parse this guard does not list; a file the installation ships is read through '
        f'path_utils.read_installation_file, anything else is listed here with what it parses: {found}'
    )


def test_every_listed_json_parse_is_still_there():
    # A listed def that moved or was renamed would leave the list naming nothing.
    trees = dict(_every_python_module())
    gone = sorted(
        f'{rel_path} {qualname}'
        for rel_path, qualname in _JSON_PARSERS
        if rel_path not in trees
        or qualname not in {q for q, _lineno in _json_parses(trees[rel_path])}
    )
    assert gone == [], f'listed JSON parses that are no longer there: {gone}'
