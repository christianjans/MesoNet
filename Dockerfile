FROM ubuntu:20.04
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install \
  clang \
  curl \
  ffmpeg \
  git \
  libsm6 \
  libxext6 \
  python3 \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-tk \
  python3-wheel \
  sudo \
  x11-apps
RUN mkdir repo
WORKDIR /repo

COPY . .

RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install 'deeplabcut[tf]'
RUN sudo python3 setup.py install

ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1
