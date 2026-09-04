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

from modules.lumascope_api import Lumascope
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
    # Create scope in simulate mode
    scope = Lumascope(simulate=True)
    print('Scope initialized (simulate=True)')

    # Real settings: the documented loader reads data/current.json (falling
    # back to the shipped template) and validates it. A hand-built dict has
    # to carry at least 'frame' and 'objective_id', or the bring-up refuses.
    import modules.settings_init as settings_init
    from lvp_logger import logger

    settings_init.load_lvp_settings(logger, '.')
    settings = settings_init.settings
    settings['live_folder'] = str(pathlib.Path('./capture').resolve())

    # Create a ScopeSession -- the GUI-independent state container. We built
    # the scope ourselves, so the bring-up is ours: configure it from the
    # settings, then start the camera feed (connect() leaves the camera
    # configured but not grabbing). A session whose scope the factory built
    # gets both steps for free.
    session = ScopeSession.create(settings=settings, scope=scope)
    session.configure_scope()
    scope.imaging.start_streaming()
    session.start_executors()
    print('Session created, scope configured, executors started')

    # Create a ProtocolRunner from the session
    runner = session.create_protocol_runner()

    # Build the protocol configuration
    config = build_protocol_config()
    print('\nProtocol config:')
    print(f'  Positions: {len(config["positions"])}')
    print(f'  Channels: {list(config["layer_configs"].keys())}')
    print(f'  Period: {config["period"]}')
    print(f'  Duration: {config["duration"]}')

    # NOTE: Creating a Protocol from a config dict requires a tiling
    # configurations file. For this example, we show the setup without
    # actually executing, since Protocol.from_config() depends on data
    # files that may not be present in all environments.
    #
    # In a real application with the full LumaViewPro installation:
    #
    #   from modules.protocol import Protocol
    #   tiling_file = pathlib.Path("data/tiling.json")
    #   protocol = Protocol.from_config(config, tiling_configs_file_loc=tiling_file)
    #
    #   # The image capture config is REQUIRED: it states the run's image
    #   # mode (bit depth + on-disk encoding) explicitly -- there is no
    #   # silent default. Modes: '8bit', '12bit_scientific', '12bit_scaled',
    #   # '12bit_false_color_rgb'.
    #   capture_config = runner.build_image_capture_config(image_mode="8bit")
    #
    #   # Run a single scan (captures all positions/channels once)
    #   runner.run_single_scan(
    #       protocol=protocol,
    #       sequence_name="my_scan",
    #       parent_dir=pathlib.Path("./output"),
    #       image_capture_config=capture_config,
    #   )
    #
    #   # Monitor progress
    #   print(f"Running: {runner.is_running()}")
    #   print(f"Output dir: {runner.run_dir()}")
    #
    #   # Wait for completion (blocks until done)
    #   completed = runner.wait_for_completion(timeout=300)
    #   print(f"Completed: {completed}")
    #
    #   # For a full timed protocol (repeats scans over duration):
    #   runner.run_protocol(
    #       protocol=protocol,
    #       sequence_name="my_protocol",
    #       image_capture_config=capture_config,
    #   )
    #
    #   # To abort a running protocol:
    #   runner.abort()

    print('\nProtocol setup complete (not executed in simulate-only example)')
    print('See comments in source for full execution flow')

    # Clean up
    runner.shutdown()
    session.shutdown_executors()
    scope.disconnect()
    print('Scope disconnected')


if __name__ == '__main__':
    main()
