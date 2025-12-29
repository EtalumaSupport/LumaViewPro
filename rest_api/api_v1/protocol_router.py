from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_settings, get_sequenced_capture_executor, get_source_path, get_protocol_callbacks, get_image_capture_config
from rest_api.api_v1.api_utils import model_param_description
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING
from modules.protocol import Protocol
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
import os
import pathlib
import datetime

if TYPE_CHECKING:
    from ...modules.sequenced_capture_executor import SequencedCaptureExecutor

protocol_router = APIRouter(prefix="/protocol", tags=['Protocol'])

@protocol_router.get("", description='List all saved protocols')
def list_protocols(settings:dict = Depends(get_settings)):
    if 'live_folder' in settings:
        protocols = []
        path = settings['live_folder']
        for filename in os.listdir(path):
            filename:str
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.tsv'):
                protocols.append(filename)
        return protocols
    raise HTTPException(status_code=500, detail="Could not list protocols")

class ProtocolParameters(BaseModel):
    protocol_name: str = Field(..., description="Name of protocol to run")

@protocol_router.post("/run", description=model_param_description("Runs an existing saved protocol", ProtocolParameters))
def run_protocol(protocol_parameters:ProtocolParameters,
                 settings:dict = Depends(get_settings),
                 source_path:str = Depends(get_source_path),
                 callbacks:dict = Depends(get_protocol_callbacks),
                 image_capture_config:dict = Depends(get_image_capture_config),
                 sequenced_capture_executor:"SequencedCaptureExecutor" = Depends(get_sequenced_capture_executor)):
    if 'live_folder' in settings:
        path = os.path.join(settings['live_folder'],protocol_parameters.protocol_name)
        if not path.endswith('.tsv'):
            path += '.tsv'
        if os.path.exists(path):
            protocol = Protocol.from_file(
                file_path=path,
                tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            )

            autogain_settings = settings['protocol']['autogain'].copy()
            autogain_settings['max_duration'] = datetime.timedelta(seconds=autogain_settings['max_duration_seconds'])
            del autogain_settings['max_duration_seconds']

            sequenced_capture_executor.run(protocol=protocol,
                                           run_trigger_source='rest_api',
                                           run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
                                           sequence_name=os.path.basename(path),
                                           image_capture_config=image_capture_config,
                                           autogain_settings=autogain_settings,
                                           parent_dir=pathlib.Path(settings['live_folder']).resolve() / "ProtocolData",
                                           enable_image_saving=True,
                                           separate_folder_per_channel=True,
                                           callbacks=callbacks,
                                           max_scans=None,
                                           return_to_position=None,
                                           disable_saving_artifacts=False,
                                           save_autofocus_data=False,
                                           update_z_pos_from_autofocus=False,
                                           leds_state_at_end="off",
                                           video_as_frames=settings['video_as_frames']
                                           )
            return{"message":"Protocol started"}

    raise HTTPException(status_code=500, detail="Could not run protocol")