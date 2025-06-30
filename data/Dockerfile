FROM ubuntu:25.10 as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*


ARG ADMETE_REPO="https://github.com/orbita2d/admete.git"
ARG ADMETE_TAG=v1.8.0pre-2025-06-02

RUN git init && \
  git remote add origin ${ADMETE_REPO} && \
  git fetch --depth=1 origin tag ${ADMETE_TAG} && \
  git checkout ${ADMETE_TAG} && \
  git lfs pull && \
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON -DWITH_TESTS=OFF -DWITH_BINDINGS=ON && \
  cmake --build build --config Release && \
  chmod +x build/admete

FROM sbtscala/scala-sbt:eclipse-temurin-alpine-17.0.15_6_1.11.0_3.7.0@sha256:1244dc3e858321e45fdf781abbef797307def4d45eeaa744adb071e750195708

RUN apk add --no-cache \
    gcompat \
    libc6-compat \
    libstdc++ \
    libgcc

WORKDIR /app
COPY project/ project/
COPY build.sbt .
RUN sbt update

COPY src/ src/
RUN sbt stage

COPY --from=builder /build/ /admete-build/

ENV PGN_DIR=/chess/games/lichess_db_standard_rated_2023-03.pgn.zst
ENV OUTPUT_DIR=/chess/training/data
ENV POSITIONS=1000000
CMD target/universal/stage/bin/admete-nn-data $POSITIONS /admete-build/bindings/libadmete.so $PGN_DIR $OUTPUT_DIR