# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A retired setting is one row of ``settings_init._RETIRED_PATHS``, never a block of code.

Each retirement was written as its own ``if ... pop(...) is not None`` block
in ``normalize_loaded_settings``, 16 blocks over 19 paths (two of them loops
over a tuple of keys), until they became one list and one routine. This
guard refuses a key removed anywhere in ``settings_init.py`` outside the
functions named in ``_REMOVERS``, so retiring a setting can only be a row;
each of the others removes a key as part of something else. It also refuses
a retired path the template still ships: a key removed at every load and
merged back in from the template is both.
"""

from __future__ import annotations

import ast

import _lumascope.modules.common_utils as common_utils
from _lumascope.modules import settings_init
from tests.ast_seams import MemberNames, parse_module, walk_defs
from tests.settings_fixtures import shipped_template

_MODULE = 'src/_lumascope/modules/settings_init.py'

# The functions that remove a key, each for its own reason.
_REMOVERS = MemberNames(
    _MODULE,
    {
        # The one routine over the retired list.
        '_remove_retired_paths',
        # A key renamed: the old name moves to the new.
        '_migrate_renamed_keys',
        'migrate_video_settings_dict',
        # A channel's block under an older release's name, folded to its key.
        'take_old_channel_blocks',
        'fold_old_channel_blocks',
        # The template's defaults block, consumed as each channel is filled.
        '_give_every_channel_a_block',
    },
)


def _removes_a_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Delete) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ('pop', 'popitem')
    )


def test_no_setting_is_retired_by_a_block_of_its_own():
    removals: dict[int, str] = {}
    for qualname, fn in walk_defs(parse_module(_MODULE).body):
        for node in ast.walk(fn):
            if _removes_a_key(node):
                # The outermost def is walked first, and owns the line.
                removals.setdefault(node.lineno, qualname)
    offenders = sorted(
        (line, owner) for line, owner in removals.items() if owner.split('.')[0] not in _REMOVERS
    )
    assert not offenders, (
        f'A key removed in {_MODULE} outside its removers (line, function): {offenders}; '
        'a retired setting is a row of _RETIRED_PATHS'
    )


def test_no_retired_setting_is_shipped():
    template = shipped_template()
    shipped = []
    for path in settings_init._RETIRED_PATHS:
        first, *rest = path.split('.')
        tops = common_utils.get_channels() if first == '*' else [first]
        for top in tops:
            holder: object = template
            for segment in (top, *rest):
                holder = holder.get(segment, KeyError) if isinstance(holder, dict) else KeyError
            if holder is not KeyError:
                shipped.append(f'{top}.{".".join(rest)}' if rest else top)
    assert not shipped, f'Retired settings the template still ships: {shipped}'
