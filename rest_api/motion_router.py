from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_config import get_scope
from lumascope_api import Lumascope
from enum import Enum

motion_router = APIRouter(prefix="/move", tags=['Motion'])

class Axis(str, Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    T = "T"

@motion_router.post("/absolute")
async def move_absolute(axis: Axis, pos: float, overshoot_enabled: bool = True,
                        ignore_limits: bool = False, scope:Lumascope = Depends(get_scope)):
    scope.move_absolute_position(axis, pos, False, overshoot_enabled, ignore_limits)
    success = await scope.async_wait_until_finished_moving()
    if not success:
        raise HTTPException(status_code=500, detail="Could not complete the operation")
    return {"message":"Movement completed"}

@motion_router.post("/relative")
async def move_relative(axis: Axis, um: float, overshoot_enabled: bool = True, scope:Lumascope = Depends(get_scope)):
    scope.move_relative_position(axis, um, False, overshoot_enabled)
    success = await scope.async_wait_until_finished_moving()
    if not success:
        raise HTTPException(status_code=500, detail="Could not complete the operation")
    return {"message":"Movement completed"}

@motion_router.get("/position")
def get_position(scope:Lumascope = Depends(get_scope)):
    return scope.get_current_position()