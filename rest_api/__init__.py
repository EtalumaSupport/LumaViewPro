from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/")
def root():
    return "Welcome to the LumaViewPro REST API!"