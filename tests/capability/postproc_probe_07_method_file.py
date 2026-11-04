"""CAPABILITY: Save a cell-count method to a file, and load one back.

GUI equivalent: Object Analysis popup > Save Method As / Load Method
(CellCountControls.save_method_as and load_method_from_file, reached from
ui/file_dialogs.py's file choosers).

Headless route: session.post_processing's default_cell_count_method,
save_cell_count_method and load_cell_count_method -- the members the panel
calls. A script starts from the panel's default, saves it, loads it back
unchanged, and is refused a file whose method the count cannot use.
"""

import json
import os
import pathlib
import sys
import tempfile

import harness


def main() -> int:
    import dataclasses

    from modules.exceptions import PostProcessingRefusedError

    scratch = pathlib.Path(os.environ.get('LVP_CAPABILITY_SCRATCH') or tempfile.mkdtemp())
    path = scratch / 'cell_count_method.json'

    session, _live = harness.make_session('method_file')
    post = session.post_processing
    method = dataclasses.replace(post.default_cell_count_method(), pixels_per_um=2.6)
    post.save_cell_count_method(method, path)
    loaded = post.load_cell_count_method(path)
    harness.check(
        'a script can save a cell-count method and load it back unchanged',
        loaded == method,
        f'loaded={loaded}',
    )

    bad = scratch / 'bad_method.json'
    saved = json.loads(path.read_text())
    saved['context']['pixels_per_um'] = 0
    bad.write_text(json.dumps(saved))
    try:
        post.load_cell_count_method(bad)
        refused = None
    except PostProcessingRefusedError as e:
        refused = e
    harness.check(
        'a method file the count cannot use is refused, naming the file',
        refused is not None and str(bad) in str(refused),
        str(refused),
    )
    session.shutdown()
    return harness.report()


if __name__ == '__main__':
    sys.exit(main())
