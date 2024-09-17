import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from fastapi_sqlalchemy import DBSessionMiddleware
from fastapi_versioning import VersionedFastAPI
from hrag.routers import conversations, documents, tenants
from settings import app_settings

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())

app = FastAPI(
    title="Hybrid RAG API",
    description="Hybrid RAG API",
    version="0.1",
)

# register sub router for tenants
tenants.include_router(conversations, tags=["conversations"], prefix="/{tenant}")
tenants.include_router(documents, tags=["documents"], prefix="/{tenant}")

app.include_router(tenants)
add_pagination(app)
app = VersionedFastAPI(
    app, version_format="{major}", prefix_format="/api/v{major}", enable_latest=True
)
app.add_middleware(
    DBSessionMiddleware, db_url=app_settings.postgresql_url, commit_on_exit=True
)


origins = [
    app_settings.client_location,
]
if app_settings.debug:
    origins = ["*"]

logger.info(f"Allow CORS from {', '.join(origins)}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
