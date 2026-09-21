"""Probe 4 -- every destination a file dialog can set, from a script.

GUI entry points (all in ui/lumaviewpro.kv, dispatched by the four
`choose(context)` tables in ui/file_dialogs.py):
  live_folder                     kv:202  -> file_dialogs.py:597/649
  load_protocol                   kv:1103 -> file_dialogs.py:469/504
  saveas_protocol                 kv:1121 -> file_dialogs.py:693/721
  apply_video_gen_to_folder       kv:1642 -> file_dialogs.py:597/649
  apply_stitching_to_folder       kv:1729,1742
  apply_zprojection_to_folder     kv:1766
  apply_composite_gen_to_folder   kv:1820
  choose_quick_enhance_target     kv:1844 -> file_dialogs.py:546/567
  load_graphing_data              kv:3648
  save_graph                      kv:3847
  load_cell_count_input_image     kv:3888
  saveas_cell_count_method        kv:4172
  load_cell_count_method          kv:4184
  apply_cell_count_method_to_folder kv:4195
"""

import pathlib
import sys

import harness as _common

s, live = _common.make_session()
try:
    # --- live_folder ------------------------------------------------------
    newdir = _common.SCRATCH / 'chosen_live'
    s.update_settings('live_folder', str(newdir))
    _common.ok(
        'live_folder set through Session', s.get_settings_snapshot()['live_folder'] == str(newdir)
    )
    _common.void(
        'live_folder creates/validates the directory',
        newdir.exists(),
        'the API stores a destination it never validates, so a run '
        'discovers the bad path at save time',
    )
    s.update_settings('live_folder', '/definitely/not/a/real/place/\x00bad')
    print('live_folder after a nonsense path:', repr(s.get_settings_snapshot()['live_folder']))
    s.update_settings('live_folder', str(live))

    # --- protocol load / save --------------------------------------------
    print('scope.protocols.load_protocol:', hasattr(s.scope.protocols, 'load_protocol'))
    proto = s.scope.protocols.create_protocol(config=None) if False else None
    from modules.protocol import Protocol

    print('Protocol.to_file present     :', hasattr(Protocol, 'to_file'))
    try:
        s.scope.protocols.load_protocol('/no/such/protocol.tsv')
        _common.ok('missing protocol refused', False)
    except Exception as e:
        _common.ok('missing protocol refused', True, f'{type(e).__name__}: {e}')

    # --- what the Session offers for each remaining dialog destination ----
    wanted = {
        'apply_stitching_to_folder': ('run_stitcher', 'stitch'),
        'apply_composite_gen_to_folder': ('run_composite_gen', 'composite'),
        'apply_video_gen_to_folder': ('run_video_gen', 'video'),
        'apply_zprojection_to_folder': ('run_zprojection', 'zprojection'),
        'apply_cell_count_method_to_folder': ('apply_cell_count', 'cell_count'),
        'load_cell_count_method': ('load_cell_count_method', 'cell_count'),
        'saveas_cell_count_method': ('save_cell_count_method', 'cell_count'),
        'load_cell_count_input_image': ('set_cell_count_source', 'cell_count'),
        'choose_quick_enhance_target': ('quick_enhance', 'enhance'),
        'load_graphing_data': ('load_graphing_data', 'graph'),
        'save_graph': ('save_graph', 'graph'),
    }
    session_names = {n.lower() for n in dir(s)}
    api_names = set()
    for sub in ('imaging', 'motion', 'illumination', 'io', 'protocols', 'diagnostics'):
        obj = getattr(s.scope, sub, None)
        if obj is not None:
            api_names |= {f'{sub}.{n}'.lower() for n in dir(obj)}
    for ctx, (_name, token) in wanted.items():
        hits = [n for n in session_names | api_names if token in n and not n.startswith('_')]
        print(f'{ctx:35s} session/API surface matching {token!r}: {hits or "NONE"}')

    # --- are the post-processors themselves headless-capable? -------------
    # (module level, below the API -- shows the implementation is real, and
    # that the missing piece is the API entry point, not the work.)
    from modules.zprojector import ZProjector

    empty = _common.SCRATCH / 'empty_folder'
    empty.mkdir(exist_ok=True)
    res = ZProjector(has_turret=False).load_folder(
        path=empty,
        tiling_configs_file_loc=pathlib.Path('data/tiling.json'),
        popup=None,
        announce=False,
    )
    print('ZProjector.load_folder on an empty folder (no Kivy) ->', res)
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
