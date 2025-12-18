from fastapi import APIRouter
from .motion_router import motion_router
from .protocol_router import protocol_router
from .status_router import status_router

api_v1_router = APIRouter(prefix="/v1", tags=['v1'])

api_v1_router.include_router(motion_router)
api_v1_router.include_router(protocol_router)
api_v1_router.include_router(status_router)