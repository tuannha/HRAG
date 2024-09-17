from fastapi import APIRouter

conversations = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    responses={404: {"description": "URL not found"}},
)
documents = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "URL not found"}},
)
tenants = APIRouter(
    prefix="/tenants",
    tags=["tenant"],
    responses={404: {"description": "URL not found"}},
)
