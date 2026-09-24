"""CAPABILITY: Export enhanced (derived) copies of a capture, file or folder.

GUI equivalent: Post-Processing > Enhance > Image / Folder, i.e.
ui/post_processing.py:56 QuickEnhanceControls.set_source_file and
ui/post_processing.py:59 set_source_folder.

Headless route under test: modules.quick_enhance.QuickEnhancer.export_file
and .export_folder, over a folder a real headless capture produced.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder


def main() -> int:
    live = probe_dir('enhance')
    with headless_session(live, acquiring=('BF',)) as (session, runner):
        session.scope.motion.move_absolute('X', 40000.0)
        session.scope.motion.move_absolute('Y', 30000.0)
        session.scope.motion.move_absolute('Z', 1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        here = session.get_current_plate_position()
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_enhance',
            positions=[{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2

    from modules.quick_enhance import QuickEnhancer, QuickEnhanceSettings

    enhancer = QuickEnhancer()
    settings = QuickEnhanceSettings()
    source = sorted(folder.glob('*.tiff'))[0]

    file_result = enhancer.export_file(source, settings)
    print('export_file result:', file_result)

    progress = []
    folder_result = enhancer.export_folder(
        folder,
        settings,
        progress_callback=lambda done, total, path: progress.append((done, total)),
    )
    print('export_folder result:', folder_result)
    print('output_folder:', enhancer.output_folder(folder_result))
    print('progress ticks:', progress)
    ok = bool(file_result) and folder_result.get('status')
    print('PROBE RESULT:', 'SUCCESS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
