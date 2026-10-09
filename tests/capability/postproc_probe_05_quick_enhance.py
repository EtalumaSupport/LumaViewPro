"""CAPABILITY: Export enhanced (derived) copies of a capture, file or folder.

GUI equivalent: Post-Processing > Enhance > Image / Folder, i.e.
QuickEnhanceControls.set_source in ui/post_processing.py.

Headless route under test: session.post_processing.enhance on one
image and on the folder, over a folder a real headless capture produced.
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

        from modules.exceptions import CaptureError

        source = sorted(folder.glob('*.tiff'))[0]
        progress = []
        try:
            file_result = session.post_processing.enhance(source)
            print('file result:', file_result)
            folder_result = session.post_processing.enhance(
                folder, on_progress=lambda percent, text: progress.append(text)
            )
        except CaptureError as e:
            print('enhance outcome:', type(e).__name__, e)
            print('PROBE RESULT: FAIL')
            return 1
        print('folder result:', folder_result)
        print('output_folder:', folder_result.output_folder)
        print('progress ticks:', progress)
        print('PROBE RESULT: SUCCESS')
        return 0


if __name__ == '__main__':
    sys.exit(main())
