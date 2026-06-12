# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Guard: no `.tostring()` in production source (removed NumPy API).

numpy.ndarray.tostring() (and PIL Image.tostring()) were removed in NumPy 2.0 /
Pillow 9; calling them raises AttributeError at runtime. ui/image_utils_kivy's
image_to_texture crashed the whole app on a cell-count image load because of one
such call. The replacement is .tobytes(). This guard scans production source so
a reintroduction anywhere is caught at test time, not by a field crash.
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DIRS = ('ui', 'modules', 'drivers', 'deconvolution', 'stitching_blending')


def test_no_tostring_calls_in_production_source():
    offenders = []
    for d in _DIRS:
        for path in (_ROOT / d).rglob('*.py'):
            text = path.read_text(encoding='utf-8', errors='ignore')
            for i, line in enumerate(text.splitlines(), 1):
                if '.tostring(' in line:
                    offenders.append(f'{path.relative_to(_ROOT)}:{i}: {line.strip()}')
    assert not offenders, (
        '.tostring() is removed in NumPy 2.0 -- use .tobytes():\n' + '\n'.join(offenders)
    )
