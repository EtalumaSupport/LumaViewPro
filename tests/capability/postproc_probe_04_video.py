"""CAPABILITY: Build a video from a time-lapse capture folder.

GUI equivalent: Post-Processing > Create AVI/Video > Run, i.e.
ui/post_processing.py:459 VideoCreationControls.run_video_gen.

Headless route under test: ScopeSession assembles the config, the runner
runs a short multi-scan time-lapse, then
modules.video_builder.VideoBuilder.build_from_folder encodes it.
"""

import sys

from harness import headless_session, probe_dir
from runfolder import run_protocol_folder


def main() -> int:
    live = probe_dir('video')
    # A 1 s period over a 10 s duration: several time points at one
    # position, which is what a video needs and a single scan cannot give.
    timelapse = {'period': 1.0 / 60.0, 'duration': 10.0 / 3600.0, 'labware': '96 well microplate'}
    with headless_session(live, acquiring=('BF',), protocol=timelapse) as (session, runner):
        session.scope.motion.move_absolute('X', 40000.0)
        session.scope.motion.move_absolute('Y', 30000.0)
        session.scope.motion.move_absolute('Z', 1000.0)
        session.scope.motion.wait_until_finished_moving(timeout_s=60)
        here = session.get_current_plate_position()
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_video',
            positions=[{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}],
            single_scan=False,
        )
        print('run outcome:', outcome)
        print('folder:', folder)
        if folder is None:
            print('PROBE RESULT: no run folder produced')
            return 2
        print('tiffs:', sorted(p.name for p in folder.rglob('*.tif*')))

        from modules.video_builder import VideoBuilder

        builder = VideoBuilder(has_turret=session.scope.capabilities.has_turret)
        result = builder.build_from_folder(
            path=folder,
            tiling_configs_file_loc=session.scope.protocols.tiling_configs_path(),
            popup=None,
            frames_per_sec=None,
            enable_timestamp_overlay=True,
        )
        print('video result:', result)
        videos = sorted(str(p.relative_to(folder)) for p in folder.rglob('*.avi'))
        videos += sorted(str(p.relative_to(folder)) for p in folder.rglob('*.mp4'))
        print('videos:', videos)
        print('PROBE RESULT:', 'SUCCESS' if result.get('status') else 'FAIL')
        return 0 if result.get('status') else 1


if __name__ == '__main__':
    sys.exit(main())


def probe_fps_refusal(folder, session) -> None:
    """Who refuses an out-of-range playback rate?

    ui/post_processing.py:485-494 refuses fps < 1 in the widget, before
    VideoBuilder is constructed. A headless caller never passes through
    that check, so this asks what the module itself does with fps=0.
    """
    from modules.video_builder import VideoBuilder

    builder = VideoBuilder(has_turret=session.scope.capabilities.has_turret)
    try:
        result = builder.build_from_folder(
            path=folder,
            tiling_configs_file_loc=session.scope.protocols.tiling_configs_path(),
            popup=None,
            frames_per_sec=0,
            enable_timestamp_overlay=False,
        )
        print('fps=0 ->', result)
    except Exception as e:
        print('fps=0 raised ->', type(e).__name__, e)
