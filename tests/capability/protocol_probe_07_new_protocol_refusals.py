"""Probe 07 -- the refusals the New Protocol button owns.

The GUI (ui/protocol_settings.py:731-826) checks three things itself:
  * require_file_writes_idle
  * protocol.num_steps() == 0 with no channel enabled -> popup
  * _validate_objectives_in_protocol (ui/protocol_settings.py:904)
Does the API raise/return any of them?
"""

from harness import make_session, banner

session, live = make_session('newp', home=True)
try:
    import modules.config_helpers as config_helpers

    settings = session.settings
    banner('create_protocol with NO channel enabled')
    cfg = config_helpers.get_sequenced_capture_config_from_settings(
        settings,
        objective_helper=session.objective_helper,
        wellplate_loader=session.wellplate_loader,
        tiling='1x1',
        use_zstacking=False,
    )
    for lc in cfg['layer_configs'].values():
        lc['acquire'] = None
    cfg['labware_id'] = 'Center Plate'
    try:
        p = session.scope.protocols.create_protocol(input_config=cfg)
        print('NO RAISE -- steps:', p.num_steps())
    except Exception as e:
        print(f'RAISED {type(e).__name__}: {e}')

    banner('does running an empty protocol get refused at the API?')
    from modules.protocol_runner import ProtocolRunner

    runner = ProtocolRunner(session)
    try:
        out = runner.run_single_scan(
            p,
            sequence_name='empty',
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        print('NO RAISE:', out.wait(timeout_s=60))
    except Exception as e:
        print(f'RAISED {type(e).__name__}: {e} | reason={getattr(e, "reason", None)}')

    banner('is the objective-addressability rule asked on create/load?')
    import pathlib
    import modules.lumascope_api.protocols as pmod

    src = pathlib.Path(pmod.__file__).read_text()
    print(
        'create_protocol body calls the rule:',
        'refuse_unaddressable_objectives'
        in src.split('def create_protocol')[1].split('def refuse_unaddressable')[0],
    )
    print(
        'load_protocol body calls the rule:',
        'refuse_unaddressable_objectives'
        in src.split('def load_protocol')[1].split('def create_protocol')[0],
    )
finally:
    session.shutdown()
print('\nPROBE 07 DONE')
