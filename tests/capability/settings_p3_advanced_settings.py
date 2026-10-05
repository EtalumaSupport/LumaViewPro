"""Probe 3 -- every Advanced Settings row, headless.

All bindings below are in advanced_settings.py's OWN inline Builder.load_string
kv (line numbers are file-absolute); lumaviewpro.kv never mentions them.
  scope model            ui/advanced_settings.py:485 -> :300 select_scope
  acceleration slider    ui/advanced_settings.py:513 -> :330/:365
  acceleration text box  ui/advanced_settings.py:525 -> :335 acceleration_pct_text
  stimulation enable     ui/advanced_settings.py:556 -> :268
  high conversion gain   ui/advanced_settings.py:587 -> :131
  line noise reduction   ui/advanced_settings.py:608 -> :147
  video max fps          ui/advanced_settings.py:632 -> :161
  video time limit       ui/advanced_settings.py:656 -> :196
  video timestamp ovl    ui/advanced_settings.py:675 -> :225
  live view fps          ui/advanced_settings.py:700 -> :237
  protocol LED on        ui/advanced_settings.py:741 -> :256
  keep LED between steps ui/advanced_settings.py:764 -> :262
  tiling overlap         ui/advanced_settings.py:792 -> :277
  show step locations    ui/advanced_settings.py:818 -> :289
  separate folders       ui/advanced_settings.py:841 -> :231
"""

import sys

import harness as _common

from modules.exceptions import SettingRefusedError

s, live = _common.make_session()
try:
    im, mo = s.scope.imaging, s.scope.motion

    # --- camera low-noise toggles: real API setters ----------------------
    caps = s.scope.capabilities
    print(
        'caps conversion_gain / line_noise / xy_stage:',
        caps.camera_supports_conversion_gain_mode,
        caps.camera_supports_line_noise_reduction,
        caps.has_xy_stage,
    )
    for fn, good, bad in (
        (im.set_conversion_gain_mode, 'High', 'Sideways'),
        (im.set_line_noise_reduction, True, 'yes-please'),
    ):
        try:
            print(f'{fn.__name__}({good!r}) ->', fn(good))
        except Exception as e:
            print(f'{fn.__name__}({good!r}) raised {type(e).__name__}: {e}')
        try:
            print(f'{fn.__name__}({bad!r}) ->', fn(bad))
        except Exception as e:
            print(f'{fn.__name__}({bad!r}) raised {type(e).__name__}: {e}')

    # --- acceleration limit: API setter with a documented raise ----------
    try:
        mo.set_acceleration_limit(val_pct=50)
        _common.ok('acceleration 50% applied', True)
    except Exception as e:
        _common.ok('acceleration 50% applied', False, f'{type(e).__name__}: {e}')
    for bad in (0, 250, -5):
        try:
            mo.set_acceleration_limit(val_pct=bad)
            _common.void(f'acceleration {bad}% refused', False, 'accepted')
        except Exception as e:
            _common.void(f'acceleration {bad}% refused', True, f'{type(e).__name__}: {e}')
    s.set_acceleration_limit(50)
    _common.ok(
        'acceleration stored', s.get_settings_snapshot()['motion']['acceleration_max_pct'] == 50
    )

    # --- pure settings-store rows ----------------------------------------
    s.update_settings('video.max_fps', 25)
    s.update_settings('video.max_duration_seconds', 120)
    s.update_settings('video.timestamp_overlay', False)
    back = s.get_settings_snapshot()['video']
    _common.ok(
        'video limits took effect',
        back['max_fps'] == 25 and back['max_duration_seconds'] == 120,
        str(back),
    )
    # out of range: the writer owns 0..200 / 1..3600
    for path, bad in (('video.max_fps', 9999), ('video.max_duration_seconds', 999999)):
        try:
            s.update_settings(path, bad)
            _common.ok(f'{path} {bad} refused by the API', False, 'stored')
        except SettingRefusedError as e:
            _common.ok(f'{path} {bad} refused by the API', True, f'{e.reason}: {e}')

    def refused(key, bad):
        try:
            s.update_settings(key, bad)
        except SettingRefusedError as e:
            return True, f'{e.reason}: {e}'
        return False, f'stored {s.get_settings_snapshot()[key]!r} unchecked'

    # A bad value of the wrong kind, or outside a range the writer owns, is
    # refused; a number with no owned range is not (live_view_fps).
    for key, good, bad, is_refused in (
        ('live_view_fps', 15, -3, False),
        ('tiling_overlap_percent', 15.0, 999.0, True),
        ('protocol_led_on', True, 'maybe', True),
        ('keep_led_between_steps', True, None, True),
        ('show_step_locations', True, 'x', True),
        ('separate_folder_per_channel', True, 3, True),
        ('stimulation_enabled', False, 'sure', True),
    ):
        s.update_settings(key, good)
        got = s.get_settings_snapshot()[key]
        _common.ok(f'{key} took effect', got == good, f'stored {got!r}')
        was_refused, detail = refused(key, bad)
        if is_refused:
            _common.ok(f'{key} bad value {bad!r} refused', was_refused, detail)
        else:
            _common.void(f'{key} bad value {bad!r} refused', was_refused, detail)
        s.update_settings(key, good)

    # tiling overlap: is there a validator a script can reach?
    from modules.tiling_config import TilingConfig

    try:
        TilingConfig.validate_overlap_percent(999)
        _common.ok('TilingConfig refuses 999% overlap', False)
    except Exception as e:
        _common.ok('TilingConfig refuses 999% overlap', True, f'{type(e).__name__}: {e}')
    was_refused, detail = refused('tiling_overlap_percent', 999.0)
    _common.ok('the settings writer refuses 999% overlap', was_refused, detail)

    # scope model change: what does the GUI do that a script cannot?
    print(
        'session has reconfigure/select_scope?',
        hasattr(s, 'select_scope'),
        hasattr(s, 'reconfigure_for_scope'),
    )
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
