FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv python install 3.12 && uv venv $VIRTUAL_ENV --python python3.12

# Install common ML packages globally using uv
RUN uv pip install \
    jax[cuda12] \
    flax \
    optax \
    orbax \
    pyarrow \
    orbax-checkpoint \
    tensorstore \
    numpy \
    pandas \
    scipy \
    matplotlib \
    tqdm

# Set Python to run unbuffered
ENV PYTHONUNBUFFERED=1
