FROM pyrosetta

RUN mkdir /proteinFolding

COPY . /proteinFolding

RUN cd proteinFolding && \
    pip3 install -r requirements.txt && \
    rm -rf $HOME/.cache/pip && \

RUN cd proteinFolding/stable_baseline && \
    pip3 install .

RUN cd proteinFolding/gym-rosetta && \
    pip3 install .

CMD /bin/bash
