FROM python:3.11-bullseye

## set working directory
WORKDIR /app

## instal
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential cmake git -y
ADD tools /app/tools
ADD pyproject.toml /app

ADD requirements.txt /app
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install .

## define image execution
ENTRYPOINT ["tools"]
