FROM uv-base:latest

# Install build essentials and CMake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    curl \
    p7zip-full \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Copy source code
COPY CMakeLists.txt .
COPY src/ src/
COPY test/ test/
COPY include/ include/
COPY py-training/data_loading_lichess.py data_loading.py

# Build C++ components in Release mode
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF && \
    cmake --build build --config Release && \
    chmod +x build/admete

ENV GAMES_DIR=/chess/games/
ENV OUTPUT_DIR=/chess/training/
ENV ENGINE_PATH=/app/build/admete
ENV MIN_ELO=1500
ENV MAX_POSITIONS=1000000

CMD uv run data_loading.py $GAMES_DIR $OUTPUT_DIR $ENGINE_PATH $MIN_ELO $MAX_POSITIONS