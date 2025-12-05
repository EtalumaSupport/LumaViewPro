from fastapi import APIRouter
from .motion_router import motion_router

api_router = APIRouter()
api_router.include_router(motion_router)


@api_router.get("/")
def root():
    return "Welcome to the LumaViewPro REST API!"