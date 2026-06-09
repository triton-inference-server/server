# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Derivative image: start from an official Triton server image and replace only
# the tritonserver executable and libtritonserver.so. Backends, Python wheels,
# and the rest of the filesystem stay unchanged from the base image.
#
# After a local CMake install (same layout as your -DCMAKE_INSTALL_PREFIX):
#   install/bin/tritonserver
#   install/lib/libtritonserver.so
#
# Prepare artifacts (from repository root):
#   mkdir -p replace-artifacts
#   cp install/bin/tritonserver replace-artifacts/
#   cp install/lib/libtritonserver.so replace-artifacts/
#   cp /etc/odbc.ini replace-artifacts/odbc.ini   # optional if databaseIp is set (DSN unused)
#   Run `odbcinst -q -d` inside the built image to see the exact ODBC driver name for
#   optional JSON field "odbcDriverName" if the default fails.
#
# Build (from repository root; match your Triton tag, e.g. r25.03):
#   docker build \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.03-py3 \
#     -t tritonserver:25.03-custom .
#
# CPU-only base example:
#   docker build \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.03-py3-min \
#     -t tritonserver:25.03-custom-cpu .
#
# If your base image stores libtritonserver.so under lib64:
#   --build-arg TRITON_LIB_SUBDIR=lib64
#
# Ubuntu 24.04 (noble) base images: use Connector package for 24.04, e.g.:
#   --build-arg MYSQL_ODBC_DEB_VERSION=9.7.0-1ubuntu24.04
#
# Run (mount a model repository and optionally override DM config):
#   docker run --rm --gpus=all \
#     -p8000:8000 -p8001:8001 -p8002:8002 \
#     -v /path/to/model_repo:/models:ro \
#     -v /path/to/your-triton-dmconfig.json:/etc/triton-dmconfig.json:ro \
#     tritonserver:25.03-custom \
#     tritonserver --model-repository=/models
#
# For CPU-only, drop --gpus=all and use a CPU/min base image.

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.03-py3

FROM ${BASE_IMAGE}

ARG TRITON_INSTALL_PREFIX=/opt/tritonserver
ARG TRITON_LIB_SUBDIR=lib

ARG MYSQL_ODBC_DEB_VERSION=9.7.0-1ubuntu22.04
ARG MYSQL_ODBC_DEB_ARCH=amd64

# unixODBC + official MySQL Connector/ODBC .deb (libmyodbc8 is often only in Ubuntu Universe
# or missing on minimal images). Override MYSQL_ODBC_DEB_* for noble/arm64, etc.
USER root

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unixodbc \
    odbcinst; \
  DEB="mysql-connector-odbc_${MYSQL_ODBC_DEB_VERSION}_${MYSQL_ODBC_DEB_ARCH}.deb"; \
  curl -fsSL -o "/tmp/${DEB}" \
    "https://repo.mysql.com/apt/ubuntu/pool/mysql-tools/m/mysql-connector-odbc/${DEB}"; \
  apt-get install -y "/tmp/${DEB}" || apt-get -fy install; \
  rm -f "/tmp/${DEB}"; \
  rm -rf /var/lib/apt/lists/*

# Optional DSN file (only used when databaseIp is empty in triton-dmconfig.json).
COPY replace-artifacts/odbc.ini /etc/odbc.ini
RUN chmod 644 /etc/odbc.ini

# Default DM database metadata (override at runtime with -v ...:/etc/triton-dmconfig.json:ro)
COPY replace-artifacts/triton-dmconfig.json /etc/triton-dmconfig.json
RUN chmod 644 /etc/triton-dmconfig.json

# Paths relative to the build context (repository root when building with ".")
COPY replace-artifacts/tritonserver ${TRITON_INSTALL_PREFIX}/bin/tritonserver
COPY replace-artifacts/libtritonserver.so \
  ${TRITON_INSTALL_PREFIX}/${TRITON_LIB_SUBDIR}/libtritonserver.so

# Match ownership used by generated Triton Dockerfiles (triton-server uid)
RUN chown 1000:1000 \
      ${TRITON_INSTALL_PREFIX}/bin/tritonserver \
      ${TRITON_INSTALL_PREFIX}/${TRITON_LIB_SUBDIR}/libtritonserver.so \
  && chmod 755 \
      ${TRITON_INSTALL_PREFIX}/bin/tritonserver \
      ${TRITON_INSTALL_PREFIX}/${TRITON_LIB_SUBDIR}/libtritonserver.so

