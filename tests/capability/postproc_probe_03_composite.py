"""CAPABILITY: Merge a folder's per-channel captures into composite images.

GUI equivalent: Post-Processing > Composite Gen > Run, i.e.
ui/post_processing.py:382 CompositeGenControls.run_composite_gen.

Headless route under test: ScopeSession assembles the config, the runner
captures a two-channel protocol folder, then
session.post_processing.composite merges it.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder
from tests.stock_channels import LED0


def main() -> int:
    live = probe_dir('composite')
    with headless_session(live, acquiring=('BF', LED0)) as (session, runner):
        # A real plate coordinate from the API, so the probe cannot invent
        # one outside the stage's travel.
        session.scope.motion.move_xy(x_um=40000.0)
        session.scope.motion.move_xy(y_um=30000.0)
        session.scope.motion.move_z(1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        here = session.get_current_plate_position()
        print('plate position:', here)
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_composite',
            positions=[{'x': here.x_mm, 'y': here.y_mm, 'z': here.z_um, 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        print('folder:', folder)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2
        print('tiffs:', sorted(p.name for p in folder.rglob('*.tif*')))

        from modules.exceptions import CaptureError

        try:
            result = session.post_processing.composite(folder)
        except CaptureError as e:
            print('composite outcome:', type(e).__name__, e)
            # Composite groups by the run's record, not by the files on
            # disk: the rows it read are the evidence for a failed merge.
            print('live work at the merge:', session.live_work)
            print('protocol_record.tsv:')
            print((folder / 'protocol_record.tsv').read_text())
            print('PROBE RESULT: FAIL')
            return 1
        print('composite result:', result)
        print('folder after:', sorted(str(p.relative_to(folder)) for p in folder.rglob('*.tif*')))
        print('PROBE RESULT: SUCCESS')
        return 0


if __name__ == '__main__':
    sys.exit(main())
