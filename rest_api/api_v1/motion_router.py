from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_scope
from lumascope_api import Lumascope
from enum import Enum
from pydantic import BaseModel

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
    motion_type: MotionType
    axis: Axis
    um: float
    overshoot_enabled: bool = True
    ignore_limits: bool = False

@motion_router.post("")
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

@motion_router.get("/position")
def get_position(scope:Lumascope = Depends(get_scope)):
    return scope.get_current_position()