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
    s.update_settings('motion', {**s.get_settings_snapshot()['motion'], 'acceleration_max_pct': 50})
    _common.ok(
        'acceleration stored', s.get_settings_snapshot()['motion']['acceleration_max_pct'] == 50
    )

    # --- pure settings-store rows ----------------------------------------
    snap = s.get_settings_snapshot()
    video = dict(snap.get('video', {}))
    video.update(max_fps=25, max_duration_seconds=120, timestamp_overlay=False)
    s.update_settings('video', video)
    back = s.get_settings_snapshot()['video']
    _common.ok(
        'video limits took effect',
        back['max_fps'] == 25 and back['max_duration_seconds'] == 120,
        str(back),
    )
    # out of range: widget refuses 0..200 / 1..3600; does anything else?
    bad_video = dict(back)
    bad_video.update(max_fps=9999, max_duration_seconds=999999)
    s.update_settings('video', bad_video)
    _common.void(
        'video max_fps 9999 refused by the API',
        False,
        f'stored {s.get_settings_snapshot()["video"]["max_fps"]}',
    )
    s.update_settings('video', video)

    for key, good, bad in (
        ('live_view_fps', 15, -3),
        ('tiling_overlap_percent', 15, 999),
        ('protocol_led_on', True, 'maybe'),
        ('keep_led_between_steps', True, None),
        ('show_step_locations', True, 'x'),
        ('separate_folder_per_channel', True, 3),
        ('stimulation_enabled', False, 'sure'),
        ('microscope', 'LS620', 'NotAScope'),
    ):
        s.update_settings(key, good)
        got = s.get_settings_snapshot()[key]
        _common.ok(f'{key} took effect', got == good, f'stored {got!r}')
        s.update_settings(key, bad)
        stored = s.get_settings_snapshot()[key]
        _common.void(
            f'{key} bad value {bad!r} refused', stored != bad, f'stored {stored!r} unchecked'
        )
        s.update_settings(key, good)

    # tiling overlap: is there a validator a script can reach?
    from modules.tiling_config import TilingConfig

    try:
        TilingConfig.validate_overlap_percent(999)
        _common.ok('TilingConfig refuses 999% overlap', False)
    except Exception as e:
        _common.ok('TilingConfig refuses 999% overlap', True, f'{type(e).__name__}: {e}')
    s.update_settings('tiling_overlap_percent', 999)
    cfg = s.get_sequenced_capture_config()
    print(
        'run config built with 999% overlap ->',
        {k: v for k, v in cfg.items() if 'tiling' in k or 'overlap' in k},
    )
    s.update_settings('tiling_overlap_percent', 15)

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
