"""Probe 5 -- save settings, load settings, support bundle, logs zip.

GUI entry points:
  load settings (startup)  ui/microscope_settings.py:209 load_settings()
                           (called from the app bring-up, not a widget)
  save settings            the session's own save of every accepted write
                           (ScopeSession.create(settings_file=...)); no widget
  support report           ui/lumaviewpro.kv:2624 -> ui/microscope_settings.py:1287
  zip logs                 ui/lumaviewpro.kv:2631 -> ui/microscope_settings.py:1363
"""

import json
import shutil
import sys
import time

import harness as _common

out = _common.SCRATCH / 'saved_settings.json'
if out.exists():
    out.unlink()
s, live = _common.make_session(settings_file=str(out))
try:
    # --- an accepted write is saved, a window after it --------------------
    s.update_settings('video.max_fps', 17)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not out.exists():
        time.sleep(0.1)
    _common.ok('the session saved the write within the window', out.exists())
    saved = json.loads(out.read_text())
    _common.ok('the saved file carries the write', saved['video']['max_fps'] == 17)
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
    shutil.copy(_common.REPO / 'data' / 'settings.json', root / 'data' / 'settings.json')
    shutil.copy(out, root / 'data' / 'current.json')
    import modules.settings_init as settings_init
    from lvp_logger import logger

    loaded, rejected = settings_init.prepare_settings(logger, str(root), fall_back_to_template=True)
    _common.ok(
        'prepare_settings loaded the saved file',
        loaded['live_folder'] == str(live),
        f'rejected={rejected}',
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
    # Re-raised: a probe that crashed has no verdict, and exiting 0
    # here read as a pass.
    raise
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
