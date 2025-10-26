FROM ubuntu:22.04

# Install build essentials, CMake, and Git LFS
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install cutechess-cli from binary release
RUN wget https://github.com/cutechess/cutechess/releases/download/v1.3.1/cutechess-cli-1.3.1-linux64.tar.gz && \
    tar xf cutechess-cli-1.3.1-linux64.tar.gz && \
    rm -rf cutechess-cli-1.3.1-linux64* && \
    chmod +x cutechess-cli/cutechess-cli && ln cutechess-cli/cutechess-cli /bin/cutechess-cli 

# Set up app directory for our engine
WORKDIR /app

# Copy source code for the current version
COPY CMakeLists.txt .
COPY src/ src/
COPY test/ test/
COPY include/ include/

# Build development version (current)
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF && \
    cmake --build build --config Release

# Make engine executable
RUN chmod +x build/admete

# Set up directory for opponent engine
WORKDIR /opponent

# Arguments for building the opponent from a git tag
ARG REPO_URL="https://github.com/orbita2d/admete.git"
ARG GIT_TAG="v1.5.0"
ARG OPPONENT_NAME="admete-${GIT_TAG}"

# Clone the repository at the specified tag and build it
RUN git init && \
    git remote add origin ${REPO_URL} && \
    git fetch --depth=1 origin tag ${GIT_TAG} && \
    git checkout ${GIT_TAG} && \
    git lfs pull && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF && \
    cmake --build build --config Release && \
    chmod +x build/admete

# Environment variables for testing configuration
ENV ROUNDS=400
ENV CONCURRENCY=10
ENV TIME_CONTROL="30+.5"
ENV BOOK="Human.bin"

# Return to app directory
WORKDIR /app

# Updated command to use the git-tagged opponent
CMD cutechess-cli \
     -concurrency ${CONCURRENCY} \
     -rounds ${ROUNDS} \
     -engine cmd=/app/build/admete name=admete-test \
     -engine cmd=/opponent/build/admete \
     -each tc=${TIME_CONTROL} book=/chess/polyglot/${BOOK} proto=uci \
     option.SyzygyPath=/chess/tablebase \
     -recover \
     -tb /chess/tablebase/ \
     -resign movecount=5 score=500 twosided=true \
     -ratinginterval 25

