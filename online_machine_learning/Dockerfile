# syntax = docker/dockerfile:experimental
FROM python:3.9-slim-bullseye
WORKDIR /app
COPY requirements.txt requirements.txt
COPY . .
RUN apt-get update \
    && apt-get -y install libpq-dev gcc
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt
EXPOSE 80

