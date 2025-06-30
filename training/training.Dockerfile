FROM uv-base:latest

WORKDIR /app

COPY training.py training.py

ENV PYTHONUNBUFFERED=1
ENV STEPS=40000
ENV REGULARIZATION=0
ENV DATA_DIR=/chess/training/data

CMD uv run training.py ${DATA_DIR} /chess/training/checkpoints ${STEPS} --regularization ${REGULARIZATION}
