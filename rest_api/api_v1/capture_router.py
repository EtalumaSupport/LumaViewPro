from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_scope, get_settings
from rest_api.api_v1.api_utils import model_param_description
from lumascope_api import Lumascope
from enum import Enum
from pydantic import BaseModel, Field
import pathlib
import datetime

capture_router = APIRouter(prefix="/capture", tags=['Capture'])

class TailIDMode(str, Enum):
    INCREMENT = 'increment'

class OutputFormat(str, Enum):
    TIFF = 'TIFF'
    OME_TIFF = 'OME-TIFF'

class CaptureParameters(BaseModel):
    file_root: str = Field('img_', description='Prefix for saved image filename')
    append: str = Field('ms', description='Suffix for saved image filename')
    color: str = Field('BF', description="color") #TODO: Create enum for valid color options, update description
    tail_id_mode: TailIDMode | None = Field(TailIDMode.INCREMENT, description='Can be "increment" or null, will add an incrementing int at the end of the filename')
    force_to_8bit: bool = Field(True, description='Force to 8bit') #TODO: Update description
    output_format: OutputFormat = Field(OutputFormat.TIFF, description='Output format, can be "TIFF" or "OME-TIFF"')
    true_color: str = Field('BF', description="color") #TODO: Create enum for valid color options, update description
    timeout: int = Field(0, description='Timeout in seconds')
    all_ones_check: bool = Field(False, description='All ones check') #TODO: update description
    sum_count: int = Field(1, description='Sum count') #TODO: update description
    sum_delay_s: float = Field(0.0, description='Sum delay (s)') #TODO: update description


@capture_router.post("/live", description=model_param_description("Capture and save the current live image to file",CaptureParameters))
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