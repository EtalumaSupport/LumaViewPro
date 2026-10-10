# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's capture root becomes a filename prefix one way.

The capture root prefixes every image a run saves and every file
post-processing builds from them. The image writer cleaned root and step
name together, so a root of ``exp/2026:10`` saved ``exp202610_A1_BF``;
post-processing prefixed the root as stored and looked for
``exp/2026:10_A1_BF``, which is not on disk. ``Protocol.capture_prefix()``
is the one derivation, and both read it. The root itself is kept as the
person typed it.
"""

import ast

import pandas as pd

from modules.protocol import Protocol
from tests.ast_seams import REPO_ROOT
from tests.test_labware_name_resolution import _STEP_HEADER, _STEP_ROW
from tests.test_protocol_roundtrip import TILING_CONFIGS

TYPED = 'exp/2026:10'
PREFIX = 'exp202610'


def _protocol(capture_root=''):
    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(),
            'custom_step_count': 0,
            'period': None,
            'duration': None,
            'capture_root': capture_root,
            'labware_id': 'Center Plate',
        },
    )


def _file(tmp_path, capture_root):
    tsv = tmp_path / 'rooted.tsv'
    tsv.write_text(
        'LumaViewPro Protocol\n'
        'Version\t5\n'
        'Period\t30.0\n'
        'Duration\t24.0\n'
        'Labware\t6 well microplate\n'
        f'Capture Root\t{capture_root}\n'
        '\n'
        'Steps\n' + _STEP_HEADER + _STEP_ROW
    )
    return tsv


class TestTheRootAndItsPrefix:
    def test_the_root_is_kept_as_typed(self):
        protocol = _protocol()

        protocol.modify_capture_root(TYPED)

        assert protocol.capture_root() == TYPED

    def test_the_prefix_is_the_root_as_a_filename_can_hold_it(self):
        assert _protocol(TYPED).capture_prefix() == PREFIX

    def test_a_loaded_root_keeps_its_text_and_gives_the_same_prefix(self, tmp_path):
        loaded = Protocol.from_file(_file(tmp_path, TYPED), tiling_configs_file_loc=TILING_CONFIGS)

        assert (loaded.capture_root(), loaded.capture_prefix()) == (TYPED, PREFIX)

    def test_no_root_is_no_prefix(self):
        assert _protocol('').capture_prefix() == ''

    def test_post_processing_names_the_files_the_writer_saved(self):
        """The writer cleans prefix + step as one name; post-processing joins
        the prefix to the step. They agree because the prefix is clean."""
        prefix = _protocol(TYPED).capture_prefix()
        step = 'A1_BF'

        assert f'{prefix}_{step}' == Protocol.sanitize_step_name(f'{prefix}_{step}')


def _calls(path, attribute):
    tree = ast.parse((REPO_ROOT / path).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def test_both_filename_builders_read_the_prefix_and_not_the_root():
    for path in ('modules/protocol_image_writer.py', 'modules/protocol_post_processor.py'):
        assert _calls(path, 'capture_prefix'), f'{path} must name files with capture_prefix()'
        assert not _calls(path, 'capture_root'), (
            f'{path} reads capture_root(): the typed root is not a filename prefix'
        )
