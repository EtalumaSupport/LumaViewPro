"""CAPABILITY: Merge a folder's per-channel captures into composite images.

GUI equivalent: Post-Processing > Composite Gen > Run, i.e.
ui/post_processing.py:382 CompositeGenControls.run_composite_gen.

Headless route under test: ScopeSession assembles the config, the runner
captures a two-channel protocol folder, then
modules.composite_generation.CompositeGeneration.load_folder merges it.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder


def main() -> int:
    live = probe_dir('composite')
    with headless_session(live, acquiring=('BF', 'Blue')) as (session, runner):
        # A real plate coordinate from the API, so the probe cannot invent
        # one outside the stage's travel.
        session.scope.motion.move_absolute('X', 40000.0)
        session.scope.motion.move_absolute('Y', 30000.0)
        session.scope.motion.move_absolute('Z', 1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        here = session.get_current_plate_position()
        print('plate position:', here)
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_composite',
            positions=[{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}],
        )
        print('run outcome:', outcome)
        print('folder:', folder)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2
        print('tiffs:', sorted(p.name for p in folder.rglob('*.tif*')))

        from modules.composite_generation import CompositeGeneration
        import modules.config_helpers as config_helpers

        comp = CompositeGeneration(has_turret=session.scope.capabilities.has_turret)
        result = comp.load_folder(
            path=folder,
            tiling_configs_file_loc=session.scope.protocols.tiling_configs_path(),
            popup=None,
            output_format=session.settings['image_output_format']['sequenced'],
            brightness_thresholds_percent=config_helpers.get_composite_blend_thresholds(
                session.settings
            ),
        )
        print('composite result:', result)
        print('folder after:', sorted(str(p.relative_to(folder)) for p in folder.rglob('*.tif*')))
        print('PROBE RESULT:', 'SUCCESS' if result.get('status') else 'FAIL')
        return 0 if result.get('status') else 1


if __name__ == '__main__':
    sys.exit(main())
