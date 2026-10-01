"""CAPABILITY: Stitch a tiled scan folder into one mosaic per group.

GUI equivalent: Post-Processing > Stitch > Quality / Fast Preview, i.e.
ui/post_processing.py:191 StitchControls.run_stitcher.

Headless route under test: ScopeSession assembles a 2x2 tiled capture
config, the runner captures it, then session.post_processing.stitch
stitches it. No Kivy, no ui.* import.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder


def main() -> int:
    live = probe_dir('stitch')
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
            tiling='2x2',
            sequence_name='probe_stitch',
            positions=[{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        print('folder:', folder)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2
        print('tiffs:', sorted(p.name for p in folder.rglob('*.tif*')))

        from modules.exceptions import CaptureError

        try:
            result = session.post_processing.stitch(folder, mode='quality')
        except CaptureError as e:
            print('stitch outcome:', type(e).__name__, e)
            print('PROBE RESULT: FAIL')
            return 1
        print('stitch result:', result)
        print('folder after:', sorted(str(p.relative_to(folder)) for p in folder.rglob('*.tif*')))
        print('PROBE RESULT: SUCCESS')
        return 0


if __name__ == '__main__':
    sys.exit(main())
