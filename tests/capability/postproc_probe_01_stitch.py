"""CAPABILITY: Stitch a tiled scan folder into one mosaic per group.

GUI equivalent: Post-Processing > Stitch > Quality / Fast Preview, i.e.
ui/post_processing.py:191 StitchControls.run_stitcher.

Headless route under test: ScopeSession assembles a 2x2 tiled capture
config, the runner captures it, then modules.stitcher.Stitcher.load_folder
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

        from modules.stitcher import Stitcher

        stitcher = Stitcher(has_turret=session.scope.capabilities.has_turret)
        result = stitcher.load_folder(
            path=folder,
            tiling_configs_file_loc=session.scope.protocols.tiling_configs_path(),
            popup=None,
            stitching_mode=Stitcher.QUALITY_MODE,
        )
        print('stitch result:', result)
        print('folder after:', sorted(str(p.relative_to(folder)) for p in folder.rglob('*.tif*')))
        print('PROBE RESULT:', 'SUCCESS' if result.get('status') else 'FAIL')
        return 0 if result.get('status') else 1


if __name__ == '__main__':
    sys.exit(main())
