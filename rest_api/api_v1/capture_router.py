from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_scope, get_settings
from lumascope_api import Lumascope
from enum import Enum
from pydantic import BaseModel
import pathlib
import datetime

capture_router = APIRouter(prefix="/capture", tags=['Capture'])

class TailIDMode(str, Enum):
    INCREMENT = 'increment'

class OutputFormat(str, Enum):
    TIFF = 'TIFF'
    OME_TIFF = 'OME-TIFF'

class CaptureParameters(BaseModel):
    file_root: str = 'img_'
    append: str = 'ms'
    color: str = 'BF' #TODO: Create enum for valid color options
    tail_id_mode: TailIDMode | None = TailIDMode.INCREMENT
    force_to_8bit: bool = True
    output_format: OutputFormat = OutputFormat.TIFF
    true_color: str = 'BF' #TODO: Create enum for valid color options
    timeout: int = 0
    all_ones_check: bool = False
    sum_count: int = 1
    sum_delay_s: float = 0.0


@capture_router.post("/live")
def live_capture(capture_parameters:CaptureParameters,
                 settings:dict = Depends(get_settings),
                 scope:Lumascope = Depends(get_scope)):
    scope.save_live_image(
        save_folder=pathlib.Path(settings['live_folder']) / "Manual",
        file_root=capture_parameters.file_root,
        append=capture_parameters.append,
        color=capture_parameters.color,
        tail_id_mode=capture_parameters.tail_id_mode,
        force_to_8bit=capture_parameters.force_to_8bit,
        output_format=capture_parameters.output_format,
        true_color=capture_parameters.true_color,
        earliest_image_ts=None,
        timeout=datetime.timedelta(seconds=capture_parameters.timeout),
        all_ones_check=capture_parameters.all_ones_check,
        sum_count=capture_parameters.sum_count,
        sum_delay_s=capture_parameters.sum_delay_s,
        sum_iteration_callback=None,
        turn_off_all_leds_after=False,
        use_executor=False
    )
    return{"message":"Capture saved"}