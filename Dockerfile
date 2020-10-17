FROM pyrosetta

RUN pip3 install --upgrade pip
RUN mkdir /proteinFolding

COPY . /proteinFolding
WORKDIR proteinFolding

RUN pip3 install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

RUN cd stable_baseline && \
    pip3 install .

RUN pip3 install -e gym-rosetta

RUN git clone https://github.com/harmslab/pdbtools.git && \
    pip3 install -e pdbtools

CMD /bin/bash
