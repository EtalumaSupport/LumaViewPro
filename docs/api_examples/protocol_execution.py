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
from unittest.mock import MagicMock

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Mock modules not needed for headless simulate mode
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.version = "example"
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

from lumascope_api import Lumascope
from modules.scope_session import ScopeSession
from modules.protocol_runner import ProtocolRunner
from modules.protocol import Protocol


def build_protocol_config():
    """Build a protocol configuration dict.

    This defines a simple protocol that captures two channels (BF and Blue
    fluorescence) at a single position. In a real workflow, you would
    typically have multiple well positions and possibly tiling.
    """
    return {
        'labware_id': '96 Well Plate',
        'objective_id': '4x',
        'period': datetime.timedelta(minutes=5),   # Time between scans
        'duration': datetime.timedelta(hours=1),    # Total protocol duration
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
                'illumination': 100,
                'gain': 0,
                'auto_gain': False,
                'exposure': 50,
                'sum_count': 1,
                'acquire': True,
                'autofocus': False,
            },
            'Blue': {
                'color': 'Blue',
                'false_color': False,
                'illumination': 50,
                'gain': 6,
                'auto_gain': False,
                'exposure': 200,
                'sum_count': 1,
                'acquire': True,
                'autofocus': False,
            },
        },
    }


def main():
    # Create scope in simulate mode
    scope = Lumascope(simulate=True)
    print("Scope initialized (simulate=True)")

    # Build session settings (minimal for headless operation)
    settings = {
        'live_folder': str(pathlib.Path('./capture').resolve()),
        'stage_offset': {'x': 0, 'y': 0},
    }

    # Create a ScopeSession -- the GUI-independent state container
    session = ScopeSession.create(settings=settings, scope=scope)
    session.start_executors()
    print("Session created, executors started")

    # Create a ProtocolRunner from the session
    runner = session.create_protocol_runner()

    # Build the protocol configuration
    config = build_protocol_config()
    print(f"\nProtocol config:")
    print(f"  Positions: {len(config['positions'])}")
    print(f"  Channels: {list(config['layer_configs'].keys())}")
    print(f"  Period: {config['period']}")
    print(f"  Duration: {config['duration']}")

    # NOTE: Creating a Protocol from a config dict requires a tiling
    # configurations file. For this example, we show the setup without
    # actually executing, since Protocol.from_config() depends on data
    # files that may not be present in all environments.
    #
    # In a real application with the full LumaViewPro installation:
    #
    #   tiling_file = pathlib.Path("data/tiling.json")
    #   protocol = Protocol.from_config(config, tiling_configs_file_loc=tiling_file)
    #
    #   # Run a single scan (captures all positions/channels once)
    #   runner.run_single_scan(
    #       protocol=protocol,
    #       sequence_name="my_scan",
    #       parent_dir=pathlib.Path("./output"),
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
    #   runner.run_protocol(protocol=protocol, sequence_name="my_protocol")
    #
    #   # To abort a running protocol:
    #   runner.abort()

    print("\nProtocol setup complete (not executed in simulate-only example)")
    print("See comments in source for full execution flow")

    # Clean up
    runner.shutdown()
    session.shutdown_executors()
    scope.disconnect()
    print("Scope disconnected")


if __name__ == '__main__':
    main()
