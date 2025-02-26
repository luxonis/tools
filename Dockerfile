FROM python:3.11-bullseye

## Set working directory
WORKDIR /app

## Install dependencies (including required libraries)
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential cmake git -y

## Add necessary files and set permissions
ADD tools /app/tools
ADD pyproject.toml /app
ADD requirements.txt /app

## Create non-root user and set ownership of the working directory
RUN adduser --disabled-password --gecos "" --no-create-home non-root && \
    chown -R non-root:non-root /app

## Install Python dependencies
RUN pip install .

## Switch to non-root user
USER non-root

## Set PATH for the installed executable
ENV PATH="/home/non-root/.local/bin:/usr/local/bin:$PATH"

## Define image execution
ENTRYPOINT ["tools"]
