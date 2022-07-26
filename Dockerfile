FROM python:3.8-bullseye

RUN python3 -m pip install -U pip
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN python3 -m pip install -r requirements.txt
WORKDIR /app/client
ADD client/package.json /app/client/package.json
ADD client/package-lock.json /app/client/package-lock.json
RUN npm install
ADD client/public /app/client/public
ADD client/src /app/client/src
RUN npm run build
WORKDIR /app
ADD . .
RUN python3 -m pip install -r yolo/yolov5/requirements.txt
ENV RUNTIME prod
CMD ["python3", "/app/main.py"]
