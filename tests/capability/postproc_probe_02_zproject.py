"""CAPABILITY: Z-project a captured Z-stack folder into one image per position.

GUI equivalent: Post-Processing > Z-Projection > Run, i.e.
ui/post_processing.py:284 ZProjectionControls.run_zprojection.

Headless route under test: ScopeSession -> ProtocolRunner.run_zstack to
produce the folder, then session.post_processing.zproject to
project it. No Kivy, no ui.* import.
"""

import pathlib
import sys

from harness import headless_session, probe_dir


def main() -> int:
    live = probe_dir('zproject')
    zstack = {'step_size': 5.0, 'range': 15.0, 'position': 'Current Position at Center'}
    with headless_session(live, acquiring=('BF',), zstack=zstack) as (session, runner):
        # The stack sweeps +/- range/2 around the current Z, so park Z
        # clear of the lower travel limit first.
        session.scope.motion.move_absolute('Z', 1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        pending = runner.run_zstack(layer='BF', sequence_name='probe_zstack')
        outcome = pending.wait(timeout_s=180)
        print('zstack outcome:', outcome)
        run_dir = pathlib.Path(pending.run_dir) if pending.run_dir else None
        print('run_dir:', run_dir)
        # The outcome answers before the run's files drain; the projection
        # reads the run's record, so wait for the run's files.
        print('files:', pending.wait_for_files(timeout_s=60))

        # Find the run folder the z-stack produced.
        candidates = sorted(live.rglob('protocol_record.tsv'))
        print('protocol records found:', candidates)
        if not candidates:
            print('PROBE RESULT: no run folder produced')
            return 2
        folder = candidates[0].parent
        print('tiffs in folder:', sorted(p.name for p in folder.rglob('*.tif*')))

        from modules.exceptions import CaptureError

        try:
            result = session.post_processing.zproject(folder, method='Max')
        except CaptureError as e:
            print('zproject outcome:', type(e).__name__, e)
            print('PROBE RESULT: FAIL')
            return 1
        print('zproject result:', result)
        outputs = sorted(str(p.relative_to(folder)) for p in folder.rglob('*.tif*'))
        print('folder after:', outputs)
        print('PROBE RESULT: SUCCESS')
        return 0


if __name__ == '__main__':
    sys.exit(main())
