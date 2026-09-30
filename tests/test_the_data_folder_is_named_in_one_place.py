# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An installed build's per-user data folder is named in one place, from one version read.

The folder is `Documents/LumaViewPro <version>`. Three modules built that
name themselves -- the logger, the GUI's environment and `get_source_root`
-- from two readers of version.txt with different encodings. A byte-order
mark was stripped by one reader and kept by the other, so the parts of one
installed session could name two folders, and a fix to one reader reached
only one of them. Now `path_utils.read_version` is the one reader, it strips
the mark, and `path_utils.data_folder_name` is the one derivation.
"""

import ast
import pathlib

from modules.path_utils import data_folder_name, read_version

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write_version(tmp_path, raw: bytes) -> pathlib.Path:
    (tmp_path / 'version.txt').write_bytes(raw)
    return tmp_path


def test_a_byte_order_mark_does_not_reach_the_version(tmp_path):
    root = _write_version(tmp_path, b'\xef\xbb\xbf4.0.0-beta40\n2026-09-30 12:00\n')

    assert read_version(root) == ('4.0.0-beta40', '2026-09-30 12:00')


def test_the_folder_name_carries_the_version_alone(tmp_path):
    root = _write_version(tmp_path, b'\xef\xbb\xbf4.0.0-beta40\n')

    version, _ = read_version(root)

    assert data_folder_name(version) == 'LumaViewPro 4.0.0-beta40'
    assert data_folder_name(version).isascii()


def test_an_undecodable_file_reads_as_no_version(tmp_path):
    root = _write_version(tmp_path, b'\xff\xfe4.0\n')

    assert read_version(root) == ('', '')


def _names_the_folder_itself(rel_path: str) -> list[int]:
    """Lines where a module builds 'LumaViewPro <x>' itself instead of calling the helper."""
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding='utf-8'))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            first = node.values[0] if node.values else None
            if isinstance(first, ast.Constant) and first.value == 'LumaViewPro ':
                found.append(node.lineno)
    return found


def test_no_reader_of_the_folder_builds_its_name_itself():
    for rel_path in ('lvp_logger.py', 'modules/app_environment.py', 'modules/path_utils.py'):
        lines = _names_the_folder_itself(rel_path)
        if rel_path == 'modules/path_utils.py':
            assert len(lines) == 1, f'path_utils builds the name once, in data_folder_name: {lines}'
        else:
            assert lines == [], f'{rel_path} builds the folder name itself at {lines}'
