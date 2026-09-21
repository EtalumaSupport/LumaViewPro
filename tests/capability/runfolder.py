"""Produce a REAL protocol run folder headlessly, for the post-processing probes.

Everything here is a production call: the session assembles the capture
config, the API builds the Protocol, the runner runs it. No Kivy, no ui.*.
"""

import pathlib

import harness  # noqa: F401 -- imported for its sys.path / cwd side effect


def run_protocol_folder(
    live: pathlib.Path,
    session,
    runner,
    *,
    tiling: str = '1x1',
    use_zstacking: bool = False,
    sequence_name: str = 'probe',
    positions: list | None = None,
    single_scan: bool = True,
):
    import modules.config_helpers as config_helpers

    config = session.get_sequenced_capture_config(tiling=tiling, use_zstacking=use_zstacking)
    if positions is not None:
        # Keep the probe run small. The session selector always plans the
        # whole plate; explicit positions are the same key the composite and
        # z-stack lanes put into this config.
        config['positions'] = positions
    protocol = session.scope.protocols.create_protocol(input_config=config)
    capture_config = config_helpers.get_image_capture_config_from_settings(session.settings)
    launch = runner.run_single_scan if single_scan else runner.run_protocol
    pending = launch(
        protocol=protocol,
        sequence_name=sequence_name,
        parent_dir=str(live),
        image_capture_config=capture_config,
        enable_image_saving=True,
    )
    outcome = pending.wait(timeout_s=600)
    records = sorted(live.rglob('protocol_record.tsv'))
    return outcome, (records[0].parent if records else None)
