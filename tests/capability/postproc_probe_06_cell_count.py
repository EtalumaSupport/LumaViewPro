"""CAPABILITIES: object analysis (cell count).

  6a. Count objects in one image and get the annotated preview + stats.
      GUI: ui/post_processing.py:939 apply_method_to_preview_image ->
           ui/post_processing.py:1174 _regenerate_image_preview.
  6b. Apply a cell-count method to a whole folder, producing results.csv.
      GUI: ui/post_processing.py:945 apply_method_to_folder ->
           ui/post_processing.py:963 execute_apply_method_to_folder.
  6c. Obtain a DEFAULT cell-count method below ui/ (what the panel starts
      with) -- probed because 6a/6b both need a settings dict.

Headless route: modules.post_processing.PostProcessing.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder

# The method dict PostProcessing expects. Transcribed from
# ui/post_processing.py:915 CellCountControls._get_init_settings, which is
# the only place it exists -- see probe 6c below.
METHOD = {
    'context': {'pixels_per_um': 1.0, 'fluorescent_mode': True},
    'segmentation': {'algorithm': 'initial', 'parameters': {'threshold': 20}},
    'filters': {
        'area': {'min': 0, 'max': 100000},
        'perimeter': {'min': 0, 'max': 100000},
        'sphericity': {'min': 0.0, 'max': 1.0},
        'intensity': {
            'min': {'min': 0, 'max': 100},
            'mean': {'min': 0, 'max': 100},
            'max': {'min': 0, 'max': 100},
        },
    },
}


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
            positions=[{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2

    import modules.image_utils as image_utils
    from modules.post_processing import PostProcessing

    post = PostProcessing()
    source = sorted(folder.glob('*.tiff'))[0]
    image, bits = image_utils.load_pixels(source)

    # 6a -- one image
    preview, stats = post.preview_cell_count(image=image, settings=METHOD, significant_bits=bits)
    print('6a preview shape:', None if preview is None else preview.shape)
    print('6a summary:', stats['summary'])

    # 6a' -- the simulated frame is featureless, so a synthetic blob field
    # proves the count is real and not just a zero the path always returns.
    import numpy as np

    blobs = np.zeros((200, 200), dtype=np.uint16)
    for cy, cx in ((50, 50), (50, 150), (150, 50), (150, 150)):
        blobs[cy - 8 : cy + 8, cx - 8 : cx + 8] = 60000
    _, blob_stats = post.preview_cell_count(image=blobs, settings=METHOD, significant_bits=16)
    print("6a' synthetic-blob summary:", blob_stats['summary'])

    # 6b -- whole folder
    processed = list(post.apply_cell_count_to_folder(path=str(folder), settings=METHOD))
    print('6b processed:', processed)
    csv_path = folder / 'results.csv'
    print('6b results.csv exists:', csv_path.exists())
    if csv_path.exists():
        print('6b results.csv:\n' + csv_path.read_text().strip())

    # 6c -- is there a DEFAULT method below ui/?
    import importlib

    found = []
    for mod_name, attr in (
        ('modules.post_processing', 'default_cell_count_settings'),
        ('modules.cell_count', 'DEFAULT_SETTINGS'),
        ('modules.cell_count', 'default_settings'),
    ):
        mod = importlib.import_module(mod_name)
        if hasattr(mod, attr):
            found.append(f'{mod_name}.{attr}')
    print('6c default-method providers below ui/:', found or 'NONE')

    ok = preview is not None and csv_path.exists()
    print('PROBE RESULT:', 'SUCCESS (6a, 6b)' if ok else 'FAIL')
    print('PROBE RESULT 6c:', 'FAIL -- no default method below ui/' if not found else 'SUCCESS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
