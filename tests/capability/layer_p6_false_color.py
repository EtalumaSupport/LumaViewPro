"""P6: false colour -- is it settable headlessly, and does it change saved PIXELS?"""

import pathlib
import hashlib
import traceback
import numpy as np
import harness

s, live = harness.make_session('p6')
scope = s.scope
try:
    # 1) settable through the Session's settings store?
    before = s.get_settings_snapshot()['Blue']['false_color']
    lay = s.get_settings_snapshot()
    s.update_settings('Blue', {**lay['Blue'], 'false_color': not before})
    after = s.get_settings_snapshot()['Blue']['false_color']
    print('1: false_color via session.update_settings:', before, '->', after)
    print(
        '1: session.get_layer_configs sees:', s.get_layer_configs(['Blue'])['Blue']['false_color']
    )

    # 2) does it change the bytes on disk?
    from modules.image_save import save_image

    arr = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4096).astype(np.uint16)
    outs = {}
    for fc in (False, True):
        p = save_image(
            scope,
            array=arr.copy(),
            save_folder=live,
            file_root='fc_',
            append=f'{fc}',
            channel='Blue',
            false_color_on=fc,
            save_encoding='rgb',
            output_format='TIFF',
            significant_bits=12,
        )
        outs[fc] = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
        import tifffile

        print(f'2: false_color_on={fc} -> shape on disk', tifffile.imread(p).shape, 'sha', outs[fc])
    print('2: PIXELS DIFFER WITH false_color:', outs[False] != outs[True])
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
