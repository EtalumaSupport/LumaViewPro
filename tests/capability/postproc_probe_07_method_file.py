"""CAPABILITY: Save a cell-count method to a file, and load one back.

GUI equivalent: Object Analysis popup > Save Method As / Load Method
(CellCountControls.save_method_as and load_method_from_file, reached from
ui/file_dialogs.py's file choosers).

Headless route: modules.post_processing's default_cell_count_method,
save_cell_count_method and load_cell_count_method -- the same owner the
panel calls. A script starts from the panel's default, saves it, loads it
back unchanged, and is refused a file whose method the count cannot use.
"""

import json
import os
import pathlib
import sys
import tempfile

import harness


def main() -> int:
    from modules.exceptions import PostProcessingRefusedError
    from modules.post_processing import (
        default_cell_count_method,
        load_cell_count_method,
        save_cell_count_method,
    )

    scratch = pathlib.Path(os.environ.get('LVP_CAPABILITY_SCRATCH') or tempfile.mkdtemp())
    path = scratch / 'cell_count_method.json'

    method = default_cell_count_method()
    method['context']['pixels_per_um'] = 2.6
    save_cell_count_method(method, path)
    loaded = load_cell_count_method(path)
    loaded_method = {k: v for k, v in loaded.items() if k != 'metadata'}
    harness.check(
        'a script can save a cell-count method and load it back unchanged',
        loaded_method == method,
        f'metadata={loaded.get("metadata")}',
    )

    bad = scratch / 'bad_method.json'
    saved = json.loads(path.read_text())
    saved['context']['pixels_per_um'] = 0
    bad.write_text(json.dumps(saved))
    try:
        load_cell_count_method(bad)
        refused = None
    except PostProcessingRefusedError as e:
        refused = e
    harness.check(
        'a method file the count cannot use is refused, naming the file',
        refused is not None and str(bad) in str(refused),
        str(refused),
    )
    return harness.report()


if __name__ == '__main__':
    sys.exit(main())
