"""Probe 2 -- output formats, JPG quality, video format, display prefs.

GUI entry points probed headlessly:
  live image format      ui/lumaviewpro.kv:2306 -> ui/microscope_settings.py:642
  sequenced image format ui/lumaviewpro.kv:2328 -> ui/microscope_settings.py:659
  jpg quality            ui/lumaviewpro.kv:2364 -> ui/microscope_settings.py:651
  video recording format ui/lumaviewpro.kv:2391 -> ui/microscope_settings.py:665
  scale bar              ui/lumaviewpro.kv:2499 -> ui/microscope_settings.py:673
  crosshairs             ui/lumaviewpro.kv:2523 -> ui/microscope_settings.py:686
  live histogram eq      ui/lumaviewpro.kv:2547 -> ui/microscope_settings.py:697
  show tooltips          ui/lumaviewpro.kv:2571 -> ui/microscope_settings.py:704
  bullseye               ui/lumaviewpro.kv:2600 -> ui/microscope_settings.py:508
"""

import sys

import harness as _common

s, live = _common.make_session()
try:
    snap = s.get_settings_snapshot()
    print('image_output_format   :', snap['image_output_format'])
    print('jpg_quality           :', snap.get('jpg_quality'))
    print('video_as_frames       :', snap.get('video_as_frames'))
    print('scale_bar             :', snap.get('scale_bar'))
    print('show_tooltips         :', snap.get('show_tooltips'))

    # --- live / sequenced output format ---------------------------------
    # Nested key: update_settings() writes TOP-LEVEL keys only, so a script
    # must rewrite the whole sub-dict.
    fmt = dict(snap['image_output_format'])
    fmt['live'] = 'JPG'
    fmt['sequenced'] = 'OME-TIFF'
    s.update_settings('image_output_format', fmt)
    back = s.get_settings_snapshot()['image_output_format']
    _common.ok(
        'live/sequenced format took effect',
        back['live'] == 'JPG' and back['sequenced'] == 'OME-TIFF',
        str(back),
    )

    # out-of-range at the store?
    bad = dict(back)
    bad['live'] = 'BMP-o-matic'
    try:
        s.update_settings('image_output_format', bad)
        _common.void(
            'bogus live format refused at update_settings',
            False,
            f'stored as {s.get_settings_snapshot()["image_output_format"]["live"]!r}',
        )
    except Exception as e:
        _common.void(
            'bogus live format refused at update_settings', True, f'{type(e).__name__}: {e}'
        )
    # ... and at the run-config boundary?
    try:
        cfg = s.get_sequenced_capture_config()
        print(
            'get_sequenced_capture_config with bogus live format -> built, keys:',
            sorted(cfg)[:6],
            '...',
        )
        from modules.image_mode import ImageCaptureConfig

        try:
            ImageCaptureConfig.from_image_mode(
                s.settings['image_mode'],
                output_format_live=s.settings['image_output_format']['live'],
                output_format_sequenced=s.settings['image_output_format']['sequenced'],
            )
            _common.ok('bogus live format refused at ImageCaptureConfig', False)
        except Exception as e:
            _common.ok(
                'bogus live format refused at ImageCaptureConfig', True, f'{type(e).__name__}: {e}'
            )
    except Exception as e:
        print('get_sequenced_capture_config raised:', type(e).__name__, e)
    s.update_settings('image_output_format', back)

    # --- jpg quality ------------------------------------------------------
    s.update_settings('jpg_quality', 55)
    _common.ok('jpg_quality took effect', s.get_settings_snapshot()['jpg_quality'] == 55)
    try:
        s.update_settings('jpg_quality', 500)
        _common.void(
            'jpg_quality 500 refused', False, f'stored {s.get_settings_snapshot()["jpg_quality"]}'
        )
    except Exception as e:
        _common.void('jpg_quality 500 refused', True, f'{type(e).__name__}: {e}')
    from modules.image_mode import ImageCaptureConfig

    try:
        c = ImageCaptureConfig.from_image_mode(s.settings['image_mode'], jpg_quality=500)
        _common.void(
            'jpg_quality 500 refused at ImageCaptureConfig',
            False,
            f'built with jpg_quality={c.jpg_quality}',
        )
    except Exception as e:
        _common.void(
            'jpg_quality 500 refused at ImageCaptureConfig', True, f'{type(e).__name__}: {e}'
        )
    s.update_settings('jpg_quality', 90)

    # --- video recording format ------------------------------------------
    s.update_settings('video_as_frames', True)
    _common.ok('video_as_frames took effect', s.get_settings_snapshot()['video_as_frames'] is True)
    s.update_settings('video_as_frames', 'banana')
    print("video_as_frames after 'banana':", repr(s.get_settings_snapshot()['video_as_frames']))
    s.update_settings('video_as_frames', False)

    # --- scale bar (the one pref with a real API setter) -------------------
    im = s.scope.imaging
    im.set_scale_bar(enabled=True)
    print('scale_bar_config      :', im.scale_bar_config)
    _common.ok('scale bar on through API', im.scale_bar_config.get('enabled') is True)
    im.set_scale_bar(enabled=False)
    _common.ok('scale bar off through API', im.scale_bar_config.get('enabled') is False)
    try:
        im.set_scale_bar(enabled=True, color='chartreuse-ish')
        print('scale bar bogus colour ->', im.scale_bar_config)
    except Exception as e:
        print('scale bar bogus colour raised', type(e).__name__, e)

    # --- crosshairs / bullseye / live histogram equalization --------------
    for attr in ('use_crosshairs', 'use_bullseye', 'use_live_image_histogram_equalization'):
        print(
            f'API/session attr {attr!r} present?',
            hasattr(s, attr),
            '| on scope?',
            hasattr(s.scope, attr),
        )
    print('session has scope_display?', getattr(s, 'scope_display', 'ABSENT'))

    # --- show tooltips -----------------------------------------------------
    s.update_settings('show_tooltips', False)
    _common.ok(
        'show_tooltips took effect (store only)',
        s.get_settings_snapshot()['show_tooltips'] is False,
    )
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
