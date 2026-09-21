"""Probe 01 -- protocol file round-trip, fully headless.

Create a protocol, save it to TSV, load it back, run one scan over it,
and assert the run actually produced image files on disk.
"""

import pathlib
import time

from harness import make_session, banner

session, live = make_session('roundtrip', home=True)
try:
    import modules.config_helpers as config_helpers
    from modules.protocol_runner import ProtocolRunner

    runner = ProtocolRunner(session)

    banner('1. create a protocol (no GUI)')
    cfg = config_helpers.get_sequenced_capture_config_from_settings(
        session.settings,
        objective_helper=session.objective_helper,
        wellplate_loader=session.wellplate_loader,
        tiling='1x1',
        use_zstacking=False,
    )
    # Keep the probe quick: one channel only.
    cfg['layer_configs'] = {'BF': cfg['layer_configs']['BF']}
    cfg['layer_configs']['BF']['acquire'] = 'image'
    cfg['labware_id'] = 'Center Plate'
    protocol = session.scope.protocols.create_protocol(input_config=cfg)
    print('created steps   :', protocol.num_steps())
    print('labware         :', protocol.labware())

    banner('2. save to TSV')
    path = live / 'probe_roundtrip.tsv'
    err = protocol.to_file(file_path=path)
    print('to_file error   :', err)
    print('file exists     :', path.exists(), path.stat().st_size if path.exists() else '-')

    banner('3. load it back')
    reloaded = session.scope.protocols.load_protocol(file_path=path)
    print('reloaded type   :', type(reloaded).__name__)
    print('reloaded steps  :', reloaded.num_steps())
    print('steps match     :', reloaded.num_steps() == protocol.num_steps())

    banner('4. run one scan over the reloaded protocol')
    outcome = runner.run_single_scan(
        reloaded,
        sequence_name='probe_roundtrip',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    settled = outcome.wait(timeout_s=240)
    print('status          :', getattr(settled, 'status', None))
    print('reason          :', getattr(settled, 'reason', None))
    print('message         :', getattr(settled, 'message', None))
    run_dir = runner.run_dir()
    print('run_dir         :', run_dir)

    banner('5. did it actually write files?')
    deadline = time.time() + 60
    files = []
    while time.time() < deadline:
        if run_dir and pathlib.Path(run_dir).exists():
            files = sorted(p for p in pathlib.Path(run_dir).rglob('*') if p.is_file())
            if any(p.suffix.lower() in ('.tiff', '.tif', '.png', '.jpg') for p in files):
                break
        time.sleep(0.5)
    imgs = [p for p in files if p.suffix.lower() in ('.tiff', '.tif', '.png', '.jpg')]
    print('files in run_dir:', len(files))
    for p in files[:12]:
        print('   ', p.relative_to(run_dir), p.stat().st_size)
    print('IMAGE FILES     :', len(imgs))
    print('ASSERT images>0 :', 'PASS' if imgs else 'FAIL')
finally:
    session.shutdown()
print('\nPROBE 01 DONE')
