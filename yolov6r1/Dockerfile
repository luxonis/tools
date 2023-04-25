FROM python:3.8-bullseye

RUN python3 -m pip install -U pip
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
WORKDIR /yolov6r1
ADD requirements.txt /yolov6r1/requirements.txt
RUN python3 -m pip install -r requirements.txt
ADD . .
ENV RUNTIME prod
CMD ["python3", "/yolov6r1/main.py"]
