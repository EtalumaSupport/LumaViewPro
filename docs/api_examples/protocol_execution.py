#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol execution example.

Demonstrates:
- Creating a Protocol from a configuration dict (without loading a CSV file)
- Using ScopeSession and ProtocolRunner for GUI-independent protocol execution
- Monitoring run progress and waiting for completion

This example builds a simple protocol with a few positions and channels,
then executes it through the ProtocolRunner API.
"""

import sys
import pathlib
import datetime

# Make the repo root importable when run standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# This example runs the SAME code path two ways:
#   standalone: python3 docs/api_examples/protocol_execution.py  (the real installed deps)
#   in-suite:   tests/test_api_examples.py runs main() under the heavy-dep
#               mocks the test conftest installs before collection
# The sys.path line serves the standalone form; in-suite it is a no-op.

from modules.scope_session import ScopeSession


def build_protocol_config():
    """Build a protocol configuration dict.

    This defines a simple protocol that captures two channels (BF and Blue
    fluorescence) at a single position. In a real workflow, you would
    typically have multiple well positions and possibly tiling.
    """
    return {
        'labware_id': '96 Well Plate',
        'objective_id': '4x Oly',
        'period': datetime.timedelta(minutes=5),  # Time between scans
        'duration': datetime.timedelta(hours=1),  # Total protocol duration
        'use_zstacking': False,
        'zstack_params': {'min': 0, 'max': 0, 'step': 0},
        'tiling': 'Center',
        'binning_size': 1,
        'frame_dimensions': {'width': 1920, 'height': 1200},
        'stim_config': {},
        'positions': [
            {'x': 50000, 'y': 40000, 'z': 5000, 'name': 'A1'},
        ],
        'layer_configs': {
            'BF': {
                'color': 'BF',
                'false_color': False,
                'illumination_ma': 100,
                'gain_db': 0,
                'auto_gain': False,
                'exposure_ms': 50,
                'sum_count': 1,
                'acquire': True,
                'autofocus': False,
            },
            'Blue': {
                'color': 'Blue',
                'false_color': False,
                'illumination_ma': 50,
                'gain_db': 6,
                'auto_gain': False,
                'exposure_ms': 200,
                'sum_count': 1,
                'acquire': True,
                'autofocus': False,
            },
        },
    }


def main():
    # The user's configuration, prepared as the GUI prepares it (read,
    # checked against the shipped template, repaired, merged). A root with
    # no usable settings is refused, never quietly replaced by the template.
    settings = ScopeSession.load_user_settings('.')

    # The factory is the one door to a scope: it wires the simulated
    # drivers on the installation's data folder, configures the scope from
    # the settings, starts the camera feed and the executor lanes.
    session = ScopeSession.create(settings, simulate=True)
    session.set_live_folder(str(pathlib.Path('./capture').resolve()))
    print('Session created, scope configured (simulate=True)')

    # Build the protocol configuration
    config = build_protocol_config()
    print('\nProtocol config:')
    print(f'  Positions: {len(config["positions"])}')
    print(f'  Channels: {list(config["layer_configs"].keys())}')
    print(f'  Period: {config["period"]}')
    print(f'  Duration: {config["duration"]}')

    # NOTE: Creating a Protocol from a config dict requires a tiling
    # configurations file. For this example, we show the setup without
    # actually executing, since building a protocol depends on data
    # files that may not be present in all environments.
    #
    # In a real application with the full LumaViewPro installation:
    #
    #   # The scope resolves data/tiling.json and hands the protocol its
    #   # labware and objective catalogues.
    #   protocol = session.scope.protocols.create_protocol(input_config=config)
    #
    #   # The run captures in the session's image mode
    #   # (session.set_image_mode(...) chooses it).
    #
    #   # Run a single scan (captures all positions/channels once)
    #   pending = runner.run_single_scan(
    #       protocol=protocol,
    #       sequence_name="my_scan",
    #       parent_dir=pathlib.Path("./output"),
    #   )
    #
    #   # Monitor progress through the run's handle
    #   print(f"Running: {pending.is_live}, step {pending.step_number} of {pending.num_steps}")
    #   print(f"Output dir: {pending.run_dir}")
    #
    #   # Wait for the run to end, then read HOW it ended. None means the
    #   # bound expired; otherwise status is completed/incomplete/aborted/
    #   # failed/failed_at_start and reason is the machine-readable cause;
    #   # result.captures says what was captured of what was asked for.
    #   result = pending.wait(timeout_s=300)
    #   print(f"Ended: {result.status} ({result.reason}) -- {result.message}")
    #
    #   # For a full timed protocol (repeats scans over duration):
    #   pending = runner.run_protocol(
    #       protocol=protocol,
    #       sequence_name="my_protocol",
    #   )
    #
    #   # To stop it, through its handle; wait says when it has ended:
    #   pending.stop()
    #   pending.wait(timeout_s=60)

    print('\nProtocol setup complete (not executed in simulate-only example)')
    print('See comments in source for full execution flow')

    # Clean up: the factory built the scope, so shutdown disconnects it.
    session.shutdown()
    print('Scope disconnected')


if __name__ == '__main__':
    main()
