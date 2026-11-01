# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for tools/check_rules.py tifffile_one_importer.

Only ``modules/image_utils.py`` may import tifffile. Its two writers own a
written TIFF's codec and record, and its ``load_pixels`` returns the pixels
with their significant-bit depth; a module that imports tifffile itself
can bypass either. The rule sees every import form at any depth, and
exempts test files, whose fixtures write their inputs directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.ast_seams import tracked_files
from tools.check_rules import check_source


def _flagged(source: str, path: str) -> list[int]:
    return [v.line for v in check_source(source, path) if v.rule == 'tifffile_one_importer']


# (path, source, the lines the rule must flag)
_CASES = [
    ('modules/stitcher.py', 'import tifffile as tf\n', [1]),
    ('ui/preview.py', 'import tifffile\n', [1]),
    ('modules/zprojector.py', 'from tifffile import imwrite\n', [1]),
    ('modules/x.py', 'from tifffile.tifffile import TiffWriter\n', [1]),
    ('rest/x.py', 'import numpy, tifffile\n', [1]),
    ('lvp_logger.py', 'def f():\n    import tifffile\n', [2]),
    ('tools/x.py', 'class C:\n    def m(self):\n        from tifffile import imread\n', [3]),
    ('modules/image_utils.py', 'import tifffile as tf\n', []),
    ('tests/test_x.py', 'import tifffile as tf\n', []),
    ('modules/x.py', 'import tifffile_extras\nfrom . import tifffile\n', []),
    ('modules/x.py', 'from modules import image_utils\nimage_utils.load_pixels("a.tif")\n', []),
]


def test_the_rule_flags_every_tifffile_import_outside_its_owner():
    wrong = []
    for path, source, expected in _CASES:
        got = _flagged(source, path)
        if got != expected:
            wrong.append(f'{path} {source!r}: flagged lines {got}, expected {expected}')
    assert wrong == [], '\n'.join(wrong)


def test_no_production_module_imports_tifffile_but_image_utils():
    # The commit-time check reads only added lines; this walks the tree as it is.
    found = []
    for rel_path in tracked_files():
        source = (ROOT / rel_path).read_text(encoding='utf-8')
        found += [f'{rel_path}:{line}' for line in _flagged(source, rel_path)]
    assert found == [], (
        'only modules/image_utils.py may import tifffile, so every TIFF LVP '
        f'writes or reads goes through its writers and load_pixels: {found}'
    )
