"""P4: manual single-frame capture -- can a script get ONE frame onto disk?

Stage A: Session / lumascope_api surface only. The frame arrives; nothing on
         the API writes it, which is the void.
Stage B: modules.image_save (the Module layer the GUI reaches into directly)
         writes the file, which is the check.
"""

import pathlib
import sys
import traceback
import harness

s, live = harness.make_session('p4')
scope = s.scope
try:
    scope.imaging.start_streaming()

    # --- Stage A: is there ANY file-producing single-frame call on the API? ---
    api_objs = {
        'scope': scope,
        'imaging': scope.imaging,
        'illumination': scope.illumination,
        'motion': scope.motion,
        'io': scope.io,
        'session': s,
    }
    hits = []
    for name, o in api_objs.items():
        for attr in dir(o):
            if attr.startswith('__'):
                continue
            low = attr.lower()
            # 'snapshot' is a settings read, not a frame: it matched 'snap'
            # and reported an API save that does not exist.
            if 'snapshot' in low:
                continue
            if ('save' in low and 'image' in low) or 'snap' in low or 'save_frame' in low:
                hits.append(f'{name}.{attr}')
    print('A: API attrs that could save an image:', hits or 'NONE')

    arr = scope.imaging.capture_and_wait(force_to_8bit=False, timeout_s=5.0)
    print('A: capture_and_wait ->', type(arr).__name__, getattr(arr, 'shape', None))
    harness.check('a script can get one frame from the API', arr is not None)
    api_files = sorted(p.name for p in live.rglob('*') if p.is_file())
    print('A: files written by the API path:', api_files or 'NONE')
    harness.void(
        'the API can put one live frame on disk',
        bool(hits) or bool(api_files),
        'the manual save lives in modules.image_save, reached by the GUI directly',
    )

    # --- Stage B: the module function the GUI calls, reached directly ---
    from modules.image_save import save_live_image

    folder = live / 'Manual'
    folder.mkdir(parents=True, exist_ok=True)
    scope.illumination.led_on('Blue', 100.0, block=True)
    out = save_live_image(
        scope,
        save_folder=folder,
        file_root='live_',
        append='A1_Blue',
        channel='Blue',
        false_color_on=True,
        force_to_8bit=False,
        output_format='TIFF',
        all_ones_check=True,
        sum_count=1,
        sum_delay_s=0,
        turn_off_all_leds_after=False,
        save_encoding='right_aligned',
    )
    print('B: save_live_image ->', out)
    written = out is not None and pathlib.Path(out).is_file()
    print(
        'B: file exists:', written, 'bytes:', pathlib.Path(out).stat().st_size if written else None
    )
    harness.check('modules.image_save.save_live_image writes one frame to disk', written)
    scope.illumination.leds_off()
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    harness.check('probe completed without an unexpected raise', False)
finally:
    s.shutdown()
sys.exit(harness.report())
