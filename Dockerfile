FROM pyrosetta

RUN \
    pip install pip --upgrade && \
    pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

COPY ./setup.py ${CODE_DIR}/stable-baselines/setup.py

ENV PATH=$VENV/bin:$PATH

CMD /bin/bash
