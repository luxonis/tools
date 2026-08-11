FROM python:3.11-bullseye

## Set working directory
WORKDIR /app

## Install dependencies (including required libraries)
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 build-essential cmake git && rm -rf /var/lib/apt/lists/*

# Copy dependency descriptors and constraints used by both pip resolvers.
COPY pyproject.toml requirements.txt constraints.txt /app/

# Copy the app
COPY tools /app/tools

# Apply the constraint to normal and isolated dependency resolution.
RUN PIP_CONSTRAINT=/app/constraints.txt \
    PIP_BUILD_CONSTRAINT=/app/constraints.txt \
    pip install --no-cache-dir .

## Create non-root user and set ownership of the working directory
RUN adduser --disabled-password --gecos "" --no-create-home non-root && \
    chown -R non-root:non-root /app

## Switch to non-root user
USER non-root

## Set PATH for the installed executable
ENV PATH="/home/non-root/.local/bin:/usr/local/bin:$PATH"

## Define image execution
ENTRYPOINT ["tools"]
