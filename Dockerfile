FROM pyrosetta

RUN pip3 install --upgrade pip
RUN mkdir /proteinFolding

COPY . /proteinFolding
WORKDIR proteinFolding

RUN pip3 install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

RUN cd stable-baselines3 && \
    pip3 install -e .

RUN pip3 install -e gym-rosetta

#RUN git clone https://github.com/adbelniak/pdbtools.git && \
#    pip3 install -e pdbtools

CMD /bin/bash
