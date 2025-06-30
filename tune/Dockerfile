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

WORKDIR /app
COPY CMakeLists.txt .
COPY src/ src/
COPY test/ test/
COPY include/ include/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF && \
    cmake --build build --config Release && \
    chmod +x build/admete && \
    ln build/admete /bin/admete

ARG REPO_URL="https://github.com/orbita2d/admete.git"
ARG GIT_TAG="v1.6.0"
WORKDIR /opponent

RUN git init && \
    git remote add origin ${REPO_URL} && \
    git fetch --depth=1 origin tag ${GIT_TAG} && \
    git checkout ${GIT_TAG} && \
    git lfs pull && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF && \
    cmake --build build --config Release && \
    chmod +x build/admete && \
    ln build/admete /bin/opponent


# Environment variables for testing configuration
ENV ROUNDS=400
ENV CONCURRENCY=10
ENV TIME_CONTROL="30+.5"
ENV BOOK="Human.bin"
ENV ARGS_A=""
ENV ARGS_B=""
ENV PLAYER_A="admete-A"
ENV PLAYER_B="admete-B"
ENV ARGS_BOTH=""

# Return to app directory
WORKDIR /app

# Updated command to use the git-tagged opponent
CMD cutechess-cli \
     -concurrency ${CONCURRENCY} \
     -rounds ${ROUNDS} \
     -engine cmd=admete name=${PLAYER_A} ${ARGS_A} \
     -engine cmd=admete name=${PLAYER_B} ${ARGS_B} \
     -each tc=${TIME_CONTROL} book=/chess/polyglot/${BOOK} proto=uci ${ARGS_BOTH} \
     option.SyzygyPath=/chess/tablebase \
     -recover \
     -tb /chess/tablebase/ \
     -resign movecount=5 score=500 twosided=true \
     -pgnout games.pgn
