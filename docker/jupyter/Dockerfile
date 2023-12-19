FROM jupyter/minimal-notebook


COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"
