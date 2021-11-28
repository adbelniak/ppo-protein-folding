FROM rosetta
ARG USERNAME
ARG PASSWORD

RUN apt-get update && \
    apt-get install wget bzip2  git -y

RUN wget https://$USERNAME:$PASSWORD@graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python36.ubuntu/PyRosetta4.Release.python36.ubuntu.release-302.tar.bz2

RUN apt-get install python3-pip -y

RUN pip3 install setuptools --upgrade

RUN tar -xvjf PyRosetta4.Release.python36.ubuntu.release-302.tar.bz2 && \
    rm PyRosetta4.Release.python36.ubuntu.release-302.tar.bz2 && \
    cd PyRosetta4.Release.python36.ubuntu.release-302/setup/ && \
    python3 setup.py install