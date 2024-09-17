FROM --platform=linux/amd64 python:3.11-slim

ENV http_proxy=$http_proxy
ENV https_proxy=$http_proxy

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_CREATE=false

RUN apt -y update &&  \
    apt -y upgrade &&  \
    apt install -y build-essential && \
    apt clean && apt auto-clean

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip wheel setuptools  && pip3 install poetry
RUN poetry install --only main --no-interaction --no-ansi
RUN chmod +x docker/entrypoint.sh

EXPOSE 8000:8000

ENTRYPOINT ["docker/entrypoint.sh"]
