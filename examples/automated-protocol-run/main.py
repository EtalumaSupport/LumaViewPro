
import datetime
import importlib
import logging
import pathlib
import sys
import time


# Relative path to LVP root directory
lvp_root_path_relative = "../../"

sys.path.append(lvp_root_path_relative)

from lumascope_api import Lumascope

from modules.autofocus_executor import AutofocusExecutor
from modules.labware_loader import WellPlateLoader
from modules.protocol import Protocol
from modules.sequenced_capture_executor import SequencedCaptureExecutor
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global run_complete
run_complete = False

def main(
    protocol_tsv_file_loc: pathlib.Path
):
   
    # Create a scheduler for events (since running outside of Kivy's event loop)
    kivy_clock_module = importlib.import_module('kivy.clock')
    scheduler_config = {
        'use_kivy_clock': False,
        'scheduler': kivy_clock_module.Clock,
    }

     # Note: These values are copied from the main settings.json/current.json file
    stage_offset = {
        "x": 5500.0,
        "y": 4000.0
    }

    tiling_configs_file_loc = pathlib.Path(".") / "data" / "tiling.json"

    # Load the protocol
    my_protocol = Protocol.from_file(
        file_path=protocol_tsv_file_loc,
        tiling_configs_file_loc=tiling_configs_file_loc,
    )

    # Create the scope
    my_scope = Lumascope()

    # Reference objective ID keys from the objectives.json file
    my_scope.set_objective(objective_id="4x Oly")

    # Reference labware ID keys from labware.json file
    labware_id = "Center Plate"
    wellplate_loader = WellPlateLoader()
    labware = wellplate_loader.get_plate(plate_key=labware_id)
    my_scope.set_labware(labware)

    # Set the stage offset in the API layer
    my_scope.set_stage_offset(stage_offset)

    # Create the autofocus controller
    my_autofocus_executor = AutofocusExecutor(
        scope=my_scope,
        scheduler_config=scheduler_config,
    )

    # Create the sequenced capture executor (used for running protocols)
    my_sequenced_capture_executor = SequencedCaptureExecutor(
        scope=my_scope,
        stage_offset=stage_offset,
        autofocus_executor=my_autofocus_executor,
        scheduler_config=scheduler_config,
    )

    # Configuration for saving images
    image_capture_config = {
        'output_format': {
            'live': 'TIFF',
            'sequenced': 'TIFF',
        },
        'use_full_pixel_depth': False,
    }

    # Configuration for autogain control (from settings.json)
    autogain_settings = {
        "target_brightness": 0.3,
        "max_duration_seconds": 1.0,
        "min_gain": 0.0,
        "max_gain": 20.0
    }

    autogain_settings=convert_autogain_settings(autogain_settings=autogain_settings)

    # Top-level folder for saving captured data
    # Sub-folders will be created for each run with timestamps
    parent_dir = pathlib.Path("./capture/ProtocolData/example-automated-protocol/")

    # Callbacks that the executor can use to inform the application of certain events
    callbacks = {
        'run_complete': run_complete_cb,
    }

    logger.info("Running protocol")

    # Initiate the protocol
    my_sequenced_capture_executor.run(
        protocol=my_protocol,
        run_trigger_source=None,
        run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
        sequence_name="my_sequence",
        image_capture_config=image_capture_config,
        autogain_settings=autogain_settings,
        parent_dir=parent_dir,
        enable_image_saving=True,
        separate_folder_per_channel=False,
        callbacks=callbacks,
        max_scans=None, # Set to None to let the protocol determine max number of scans
        return_to_position=None,
        disable_saving_artifacts=False,
        leds_state_at_end="off",
        video_as_frames=True, # True will produce an image per frame, False will produce .mp4 file
    )

    while True:
        scheduler_config['scheduler'].tick()
        time.sleep(0.01)

        if run_complete:
            break


def run_complete_cb(**kwargs):
    global run_complete
    logger.info("Protocol run complete")
    run_complete = True


def convert_autogain_settings(autogain_settings: dict) -> dict:
    autogain_settings['max_duration'] = datetime.timedelta(
        seconds=autogain_settings['max_duration_seconds']
    )
    del autogain_settings['max_duration_seconds']
    return autogain_settings


if __name__ == "__main__":
    protocol_tsv_file_loc = pathlib.Path('examples/automated-protocol-run/data/example_protocol.tsv').absolute()
    main(
        protocol_tsv_file_loc=protocol_tsv_file_loc
    )
