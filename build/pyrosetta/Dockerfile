FROM rosetta
RUN wget https://$USERNAME:$PASSWORD@graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python36.mac/PyRosetta4.Release.python36.mac.release-266.tar.bz2
RUN tar -xvjf PyRosetta4.Release.python36.mac.release-266.tar.bz2 && \
    rm PyRosetta4.Release.python36.mac.release-266.tar.bz2 && \
    cd PyRosetta4.Release.python36.mac.release-266/setup/ && \
    python setup.py install
