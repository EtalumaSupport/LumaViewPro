"""P4: manual single-frame capture -- can a script get ONE frame onto disk?

Stage A: the API delivers a frame.
Stage B: session.manual_capture, the one path the GUI's Capture button also
         takes, writes it to the live folder's Manual directory.
"""

import sys
import traceback
import harness

s, live = harness.make_session('p4')
scope = s.scope
try:
    scope.imaging.start_streaming()

    # --- Stage A: the API delivers a frame ---
    arr = scope.imaging.capture_and_wait(force_to_8bit=False, timeout_s=5.0)
    print('A: capture_and_wait ->', type(arr).__name__, getattr(arr, 'shape', None))
    harness.check('a script can get one frame from the API', arr is not None)

    # --- Stage B: the Session member captures and saves one still ---
    scope.illumination.led_on('Blue', 100.0, block=True)
    paths = s.manual_capture.capture(layer='Blue', false_color_on=True).result(timeout=30)
    print('B: session.manual_capture.capture ->', paths)
    written = len(paths) == 1 and paths[0].is_file() and paths[0].parent == live / 'Manual'
    print('B: file exists:', written, 'bytes:', paths[0].stat().st_size if written else None)
    harness.check('session.manual_capture writes one frame to live/Manual', written)
    scope.illumination.leds_off()
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    harness.check('probe completed without an unexpected raise', False)
finally:
    s.shutdown()
sys.exit(harness.report())
