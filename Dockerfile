FROM --platform=linux/amd64 ubuntu:20.04
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

# # Install GTK >= 3
# RUN apt-get -y remove libgtk2.0-dev
# RUN apt-get -y install libgtk-3-dev
# # RUN apt-get -y install libgtk2.0-dev
# RUN sudo pip3 install attrdict
# RUN sudo pip3 install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04 wxPython
# # RUN sudo apt-get -y install python3-wxgtk4.0

# COPY . .

# RUN curl http://cvs.savannah.gnu.org/viewvc/*checkout*/config/config/config.guess > config.guess
# RUN chmod +x ./config.guess
# RUN bash ./config.guess

WORKDIR /tmp

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.0-3-Linux-x86_64.sh > miniconda.sh
RUN bash miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:${PATH}"
ARG PATH="/miniconda/bin:${PATH}"
RUN conda --version

RUN mkdir /dlc
WORKDIR /dlc

RUN git clone https://github.com/DeepLabCut/DeepLabCut.git .
RUN cd conda-environments && conda env create -f DEEPLABCUT.yaml
SHELL ["conda", "run", "-n", "DEEPLABCUT", "/bin/bash", "-c"]
# RUN conda init bash
# RUN conda activate DEEPLABCUT

RUN mkdir /repo
WORKDIR /repo

RUN git clone https://github.com/bf777/MesoNet.git .
RUN 

# RUN sudo pip3 install --upgrade pip
# RUN sudo pip3 install 'deeplabcut[gui,tf]'
# RUN sudo python3 setup.py install

# # ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1

# RUN sudo pip3 uninstall mesonet -y
# ENV PYTHONPATH=/repo:/repo/mesonet
