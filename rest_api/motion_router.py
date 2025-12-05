from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_config import get_scope
from lumascope_api import Lumascope

motion_router = APIRouter(prefix="/move")

@motion_router.post("/absolute")
def move_absolute(scope:Lumascope = Depends(get_scope)):
    print(scope.get_current_position())