from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_scope
from rest_api.api_v1.api_utils import model_param_description
from lumascope_api import Lumascope
from enum import Enum
from pydantic import BaseModel, Field

motion_router = APIRouter(prefix="/move", tags=['Motion'])

class Axis(str, Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    T = "T"

class MotionType(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"

class MotionParameters(BaseModel):
    motion_type: MotionType = Field(..., description='Type of positioning, either "absolute" or "relative"')
    axis: Axis = Field(..., description='Axis of motion, either "X", "Y", "Z", or "T"')
    um: float = Field(..., description='Position to move to for absolute motion, Distance to move for relative motion')
    overshoot_enabled: bool = True
    ignore_limits: bool = Field(False, description='Field ignored when using relative motion')

@motion_router.post("", description=model_param_description("Move a camera axis using absolute or relative positioning",MotionParameters))
async def move(motion_parameters: MotionParameters, scope:Lumascope = Depends(get_scope)):
    if motion_parameters.motion_type == MotionType.ABSOLUTE:
        scope.move_absolute_position(axis = motion_parameters.axis,
                                     pos = motion_parameters.um,
                                     wait_until_complete = False,
                                     overshoot_enabled = motion_parameters.overshoot_enabled,
                                     ignore_limits = motion_parameters.ignore_limits)
    elif motion_parameters.motion_type == MotionType.RELATIVE:
        scope.move_relative_position(axis = motion_parameters.axis,
                                     um = motion_parameters.um,
                                     wait_until_complete = False,
                                     overshoot_enabled = motion_parameters.overshoot_enabled)
    success = await scope.async_wait_until_finished_moving()
    if not success:
        raise HTTPException(status_code=500, detail="Could not complete the operation")
    return {"message":"Movement completed"}

@motion_router.get("/position", description="Returns the current position of each axis")
def get_position(scope:Lumascope = Depends(get_scope)):
    return scope.get_current_position()