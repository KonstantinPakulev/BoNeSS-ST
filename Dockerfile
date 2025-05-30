FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


ENV SHELL=/bin/bash

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gnupg2 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 871920D1991BC93C && \
    apt-get update && \
    apt-get install -y \
    g++ \
    cmake \
    build-essential \
    rsync \
    curl \
    git \
    wget \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libopencv-dev \
    libsuitesparse-dev \
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libboost-regex-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libcgal-dev \
    libceres-dev \
    liblz4-dev \
    libjemalloc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ceres solver
RUN \
    mkdir /source && cd /source && \
    curl -L https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz | tar xz && \
    cd ceres-solver-2.2.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_C_FLAGS=-fPIC \
             -DCMAKE_CXX_FLAGS=-fPIC \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_TESTING=OFF && \
    make -j6 install && \
    rm -rf /source/ceres-solver-2.2.0

RUN \
    cd  && \
    git clone --recursive https://github.com/laurentkneip/opengv && \
    cd opengv

# OpenGV
RUN \
    cd /source && \
    git clone --recursive https://github.com/laurentkneip/opengv && \
    cd opengv/python && rm -r pybind11 && git clone https://github.com/pybind/pybind11.git && cd .. && \
    mkdir build && cd build && \
    cmake .. -DBUILD_TESTS=OFF \
             -DBUILD_PYTHON=ON \
             -DPYTHON_INSTALL_DIR=/opt/conda/lib/python3.10/site-packages && \
    make -j6 install && \
    rm -rf /source/opengv

# Python dependencies
RUN pip install notebook \
                matplotlib \
                pandas \
                omegaconf \
                hydra-core \
                clearml \
                pytorch-lightning \
                opencv-python \
                pydegensac \
                joblib \
                h5py \
                scikit-image \
                kornia \
                e2cnn \
                tensorboard

RUN \
    cd /source && \
    git clone https://github.com/jatentaki/torch-dimcheck.git && \
    cd torch-dimcheck && \
    python setup.py install && \
    rm -rf /source/torch-dimcheck

RUN \
    cd /source && \
    git clone https://github.com/jatentaki/torch-localize.git && \
    cd torch-localize && \
    python setup.py install && \
    rm -rf /source/torch-localize

RUN \
    cd /source && \
    git clone https://github.com/jatentaki/unets.git && \
    cd unets && \
    python setup.py install && \
    rm -rf /source/unets

RUN rm -rf /source

# Configuring user
ARG UID
ARG GID

RUN addgroup --gid ${GID} --system konstantin \
 && adduser  --uid ${UID} --system \
            --ingroup konstantin \
            --home /home/konstantin \
            --shell /bin/bash konstantin

RUN chown -R konstantin:konstantin /home/konstantin

RUN usermod -aG sudo konstantin
RUN echo 'konstantin ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir "/.local"
RUN chown $UID:$GID "/.local"

USER konstantin

WORKDIR "/home/konstantin/personal/Summertime"

COPY --chown=$UID:$GID ./source ./source
COPY --chown=$UID:$GID ./train.py ./train.py
COPY --chown=$UID:$GID ./test.py ./test.py
COPY --chown=$UID:$GID ./config ./config
COPY --chown=$UID:$GID ./weights ./weights
COPY --chown=$UID:$GID ./clearml.conf /home/konstantin/clearml.conf

ENV PYTHONPATH="/home/konstantin/personal/Summertime:/home/konstantin/personal/Summertime/source/baselines/disk/disk:/home/konstantin/personal/Summertime/source/baselines/superpoint/superpoint:/home/konstantin/personal/Summertime/source/baselines/ness_st/ness_st:/home/konstantin/personal/Summertime/source/baselines/hardnet/hardnet:/home/konstantin/personal/Summertime/source/baselines/rekd/rekd:/home/konstantin/personal/Summertime/source/baselines/keynet/keynet:/home/konstantin/personal/Summertime/source/baselines/r2d2/r2d2:${PYTHONPATH}"


