FROM python:3.8-bullseye

## set working directory
WORKDIR /app

## instal
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential cmake git -y
RUN git clone --recursive https://github.com/luxonis/tools.git -b cli
ADD tools /app/tools
ADD pyproject.toml /app

RUN cd tools && pip install .
# ADD requirements.txt /app
# RUN python3 -m pip install -r requirements.txt
# RUN python3 -m pip install .

## define image execution
ENTRYPOINT ["tools"]
