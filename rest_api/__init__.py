from fastapi import APIRouter
from rest_api.api_v1 import api_v1_router


api_router = APIRouter(prefix="/api")
api_router.include_router(api_v1_router)