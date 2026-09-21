"""Probe 5 -- save settings, load settings, support bundle, logs zip.

GUI entry points:
  load settings (startup)  ui/microscope_settings.py:209 load_settings()
                           (called from the app bring-up, not a widget)
  save settings            ScopeSession.save_settings, driven by the app's
                           exit path (no widget in this slice)
  support report           ui/lumaviewpro.kv:2624 -> ui/microscope_settings.py:1287
  zip logs                 ui/lumaviewpro.kv:2631 -> ui/microscope_settings.py:1363
"""

import json
import pathlib
import shutil
import sys

import harness as _common

s, live = _common.make_session()
try:
    # --- save settings to an explicit file --------------------------------
    out = _common.SCRATCH / 'saved_settings.json'
    # The documented no_hardware refusal fires only when NOTHING is
    # connected (`camera_connected or motor_connected or led_connected`).
    # A simulated session has all three, so writing without force is the
    # contract here, not a missing guard -- the probe used to assert the
    # refusal and was reading its own precondition wrong.
    scope = s.scope
    connected = bool(scope.camera_connected or scope.motor_connected or scope.led_connected)
    try:
        s.save_settings(file=str(out))
        _common.ok(
            'save_settings writes when hardware was connected', connected, f'connected={connected}'
        )
    except Exception as e:
        _common.ok(
            'save_settings refuses only when nothing was connected',
            not connected,
            f'connected={connected}: {type(e).__name__}: {e}',
        )
    s.save_settings(file=str(out), force=True)
    _common.ok('save_settings(force=True) wrote the file', out.exists())
    saved = json.loads(out.read_text())
    _common.ok(
        'saved file carries the live values',
        saved['live_folder'] == str(live),
        saved['live_folder'],
    )

    # --- load settings from a file (the startup path, headless) -----------
    # Copy the worktree data/ so nothing touches the shared data/current.json.
    root = _common.SCRATCH / 'install_root'
    if root.exists():
        shutil.rmtree(root)
    (root / 'data').mkdir(parents=True)
    shutil.copy(
        pathlib.Path(_common.WT) / 'data' / 'settings.json', root / 'data' / 'settings.json'
    )
    shutil.copy(out, root / 'data' / 'current.json')
    import modules.settings_init as settings_init
    from lvp_logger import logger

    loaded, rejected = settings_init.prepare_settings(logger, str(root), fall_back_to_template=True)
    _common.ok(
        'prepare_settings loaded the saved file',
        loaded['live_folder'] == str(live),
        f'rejected={rejected}',
    )

    # corrupt current.json -> template fallback, not a crash
    (root / 'data' / 'current.json').write_text('{ this is not json')
    loaded2, rejected2 = settings_init.prepare_settings(
        logger, str(root), fall_back_to_template=True
    )
    _common.ok(
        'corrupt current.json falls back to the template',
        loaded2 is not None,
        f'rejected={rejected2}',
    )

    # --- support report / logs zip ---------------------------------------
    from modules.tech_support_report import TechSupportReport

    r = TechSupportReport(scope=s.scope)
    print(
        'TechSupportReport.generate signature ok:',
        hasattr(r, 'generate'),
        '| generate_logs_only:',
        hasattr(r, 'generate_logs_only'),
    )
    print(
        'Session/API surface matching "support"/"report":',
        [n for n in dir(s) if 'support' in n.lower() or 'report' in n.lower()],
    )
except Exception:
    import traceback

    traceback.print_exc()
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
