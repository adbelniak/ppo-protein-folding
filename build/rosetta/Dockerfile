FROM ubuntu:18.04
RUN apt-get -y update \
    && apt-get -y install \
    curl \
    cmake \
    default-jre \
    git \
    jq \
    python-dev \
    python-pip \
    python3-dev \
    libfontconfig1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenmpi-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o rosseta.tar.gz  -u Academic_User:Xry3x4 https://www.rosettacommons.org/downloads/academic/3.12/rosetta_src_3.12_bundle.tgz && \
    tar -xvzf rosseta.tar.gz && \
    cd rosetta_src_2020.08.61146_bundle/main/source && \
    ./scons.py -j 2 mode=release bin

