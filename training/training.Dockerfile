FROM uv-base:latest

WORKDIR /app

COPY training.py training.py

ENV PYTHONUNBUFFERED=1
ENV REGULARIZATION_LOG=0
ENV TRAIN_FILE=/chess/training/data/
ENV CHECKPOINT_PATH=/chess/training/checkpoints
ENV MLFLOW_TRACKING_URI=http://mlflow.junebug.lan:80
ENV MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Set ITERATIONS at runtime
CMD uv run training.py