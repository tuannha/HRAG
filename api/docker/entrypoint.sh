#!/bin/bash

alembic upgrade head

gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:8000" "main:app" --graceful-timeout 30000  --timeout 40000
