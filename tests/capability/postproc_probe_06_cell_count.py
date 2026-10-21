"""CAPABILITIES: object analysis (cell count).

  6a. Count objects in one image and get the annotated preview + stats.
      GUI: ui/post_processing.py:939 apply_method_to_preview_image ->
           ui/post_processing.py:1174 _regenerate_image_preview.
  6b. Apply a cell-count method to a whole folder, producing results.csv.
      GUI: ui/post_processing.py:945 apply_method_to_folder ->
           ui/post_processing.py:963 execute_apply_method_to_folder.
  6c. Obtain a DEFAULT cell-count method below ui/ (what the panel starts
      with) -- probed because 6a/6b both need a settings dict.

Headless route: session.post_processing.count_cells for the folder (6b);
session.post_processing.preview_cell_count for the one-image preview (6a).
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder

from modules.cell_count_method import CellCountMethod

# The panel's default method (6c). Its area and perimeter filters are open, so
# a simulated frame's regions are not filtered out, and it sets no scale, so
# each image is counted at the scale it states.
METHOD = CellCountMethod()


def main() -> int:
    live = probe_dir('cellcount')
    with headless_session(live, acquiring=('Blue',)) as (session, runner):
        session.scope.motion.move_absolute('X', 40000.0)
        session.scope.motion.move_absolute('Y', 30000.0)
        session.scope.motion.move_absolute('Z', 1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        here = session.get_current_plate_position()
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_cellcount',
            positions=[{'x': here.x_mm, 'y': here.y_mm, 'z': here.z_um, 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2
        csv_path = folder / 'results.csv'
        from modules.exceptions import CaptureError

        # 6b -- whole folder, through the session's member
        try:
            counted = session.post_processing.count_cells(folder, method=METHOD)
            print('6b result:', counted)
        except CaptureError as e:
            print('6b outcome:', type(e).__name__, e)
        print('6b results.csv exists:', csv_path.exists())
        if csv_path.exists():
            print('6b results.csv:\n' + csv_path.read_text().strip())

        import numpy as np

        import modules.image_utils as image_utils

        post = session.post_processing
        source = sorted(folder.glob('*.tiff'))[0]
        image, bits, _ = image_utils.load_pixels(source)

        # 6a -- one image, through the session's member
        preview = post.preview_cell_count(
            image,
            METHOD,
            significant_bits=bits,
            pixel_size_um=image_utils.read_pixel_size_um(source),
        )
        print('6a preview shape:', preview.image.shape)
        print('6a counts:', preview.counts)

        # 6a' -- the simulated frame is featureless, so a synthetic blob field
        # proves the count is real and not just a zero the path always returns.
        blobs = np.zeros((200, 200), dtype=np.uint16)
        for cy, cx in ((50, 50), (50, 150), (150, 50), (150, 150)):
            blobs[cy - 8 : cy + 8, cx - 8 : cx + 8] = 60000
        blob_counts = post.preview_cell_count(
            blobs, METHOD, significant_bits=16, pixel_size_um=None
        ).counts
        print("6a' synthetic-blob counts:", blob_counts)

    # 6c -- is there a DEFAULT method below ui/?
    import importlib

    found = []
    for mod_name, attr in (
        ('modules.cell_count_method', 'CellCountMethod'),
        ('modules.cell_count', 'DEFAULT_SETTINGS'),
        ('modules.cell_count', 'default_settings'),
    ):
        mod = importlib.import_module(mod_name)
        if hasattr(mod, attr):
            found.append(f'{mod_name}.{attr}')
    print('6c default-method providers below ui/:', found or 'NONE')

    ok = preview.image is not None and csv_path.exists()
    print('PROBE RESULT:', 'SUCCESS (6a, 6b)' if ok else 'FAIL')
    print('PROBE RESULT 6c:', 'FAIL -- no default method below ui/' if not found else 'SUCCESS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
