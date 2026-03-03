#!/bin/bash
# build_local.sh — local developer build of the ROCm Triton Inference Server.
#
# Simulates what the CI pipeline does, but without the privileged CI orchestration
# container.  Uses Docker's native BuildKit (docker buildx build) with a named
# build context to supply the pre-fetched source tree.
#
# Usage:
#   ./build_local.sh [WORK_DIR]
#
#   WORK_DIR defaults to ./triton-ci-workspace.  The pre-fetched source tree is
#   expected at WORK_DIR/src/ and the output tarball lands in WORK_DIR/artifacts/.
#
# Environment variable overrides (all optional):
#   WORK_DIR           - workspace root (overridden by positional arg if given)
#   TRITON_VERSION     - Triton version string      (default: 2.64.0)
#   BASE_IMAGE         - ROCm base image            (default: rocm/dev-ubuntu-22.04:7.2-complete)
#   BUILD_ONNXRUNTIME  - build OnnxRuntime backend  (default: OFF)
#   IMAGE_TAG          - local image tag            (default: tritonserver:<TRITON_VERSION>)
#   BUILD_TARGET       - Docker build target        (default: runtime). Use buildenv to build
#                         only the build-env image (deps + sources, no build.py); then run a
#                         container and run build.py (or ./cmake_build) manually for fast iteration.
#   SKIP_PREFETCH      - skip the prefetch step     (default: 0)
#   NO_CACHE           - pass --no-cache to buildx  (default: 0)
#   PREFETCH_SCRIPT    - path to prefetch_sources.sh (default: same dir as this script)
#
# Prerequisites:
#   - docker buildx (Docker >= 23, or BuildKit plugin installed)
#   - Sufficient disk space (~100 GB) and RAM (~32 GB) for a full build
#
# Workflow:
#   1. Run prefetch_sources.sh to clone all required sources into WORK_DIR/src/
#      (skipped when SKIP_PREFETCH=1)
#   2. Run docker buildx build, passing the source tree as the named build context
#      "local_src_dir" (used by COPY --from=local_src_dir in Dockerfile.rocm_ci)
#   3. Export the runtime image as a Docker tar in WORK_DIR/artifacts/
#
# Isolate build.py / iterate from failure (BUILD_TARGET=buildenv):
#   Build only the buildenv image (no build.py run), then run a container and execute
#   build.py (or the generated cmake_build script) by hand. Mount a host dir for
#   /tmp/tritonbuild so the build tree persists across runs.
#   Example:
#     BUILD_TARGET=buildenv WORK_DIR=/path/to/workspace SKIP_PREFETCH=1 ./build_local.sh
#     mkdir -p /path/to/workspace/tritonbuild
#     docker run -it -v /path/to/workspace/tritonbuild:/tmp/tritonbuild \
#       triton-buildenv:<TRITON_VERSION> bash
#   Inside the container, run the same build.py invocation as in Dockerfile.rocm_ci
#   (including --extra-core-cmake-arg BOOST_ROOT=/root/micromamba). To resume after
#   a failure, re-run the container with the same volume and run ./cmake_build from
#   /tmp/tritonbuild, or run make from the build subdir.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------

WORK_DIR="${1:-${WORK_DIR:-$(pwd)/triton-ci-workspace}}"

TRITON_VERSION="${TRITON_VERSION:-2.64.0}"
BASE_IMAGE="${BASE_IMAGE:-rocm/dev-ubuntu-22.04:7.2-complete}"
BUILD_ONNXRUNTIME="${BUILD_ONNXRUNTIME:-ON}"
BUILD_TARGET="${BUILD_TARGET:-runtime}"
IMAGE_TAG="${IMAGE_TAG:-tritonserver:${TRITON_VERSION}}"
SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
NO_CACHE="${NO_CACHE:-0}"

# When building only the buildenv target, tag and load that image (no output tar).
[[ "${BUILD_TARGET}" == "buildenv" ]] && IMAGE_TAG="triton-buildenv:${TRITON_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PREFETCH_SCRIPT can be overridden if prefetch_sources.sh lives elsewhere
# (e.g. when build_local.sh was installed from a separate CI tools repo).
# Default: look for it alongside this script in the same ci/ directory.
PREFETCH_SCRIPT="${PREFETCH_SCRIPT:-${SCRIPT_DIR}/prefetch_sources.sh}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[build-local] $*"; }
die()  { echo "[build-local] ERROR: $*" >&2; exit 1; }

banner() {
    log ""
    log "============================================================"
    log "  $*"
    log "============================================================"
}

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------

banner "Triton Local Build"
log "  Workspace      : ${WORK_DIR}"
log "  Triton version : ${TRITON_VERSION}"
log "  Build target   : ${BUILD_TARGET}"
log "  Base image     : ${BASE_IMAGE}"
log "  OnnxRuntime    : ${BUILD_ONNXRUNTIME}"
log "  Image tag      : ${IMAGE_TAG}"
log "  Skip prefetch  : ${SKIP_PREFETCH}"

# ---------------------------------------------------------------------------
# Step 1: Create workspace directories
# ---------------------------------------------------------------------------

banner "Step 1 — Creating workspace directories"

mkdir -p \
    "${WORK_DIR}/src" \
    "${WORK_DIR}/artifacts"

# ---------------------------------------------------------------------------
# Step 2: Prefetch sources
# ---------------------------------------------------------------------------

if [[ "${SKIP_PREFETCH}" == "1" ]]; then
    log ""
    log "SKIP_PREFETCH=1 — skipping source prefetch."
    log "(Assuming ${WORK_DIR}/src/ is already populated.)"
else
    banner "Step 2 — Prefetching sources"
    [[ -x "${PREFETCH_SCRIPT}" ]] \
        || die "prefetch_sources.sh not found or not executable: ${PREFETCH_SCRIPT}"
    BUILD_ONNXRUNTIME="${BUILD_ONNXRUNTIME}" \
    WORK_DIR="${WORK_DIR}" \
        "${PREFETCH_SCRIPT}" "${WORK_DIR}"
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SRC_ABS="$(realpath "${WORK_DIR}/src")"
ARTIFACTS_ABS="$(realpath "${WORK_DIR}/artifacts")"

DOCKERFILE="${SRC_ABS}/triton-inference-server-server/ci/Dockerfile.rocm_ci"
[[ -f "${DOCKERFILE}" ]] \
    || die "Dockerfile.rocm_ci not found at ${DOCKERFILE}. Run prefetch first (SKIP_PREFETCH=0)."

OUTPUT_TAR="${ARTIFACTS_ABS}/tritonserver-${TRITON_VERSION}.docker.tar"

# ---------------------------------------------------------------------------
# Step 3: docker buildx build
# ---------------------------------------------------------------------------

banner "Step 3 — docker buildx build"

log "  Dockerfile     : ${DOCKERFILE}"
log "  Source context : ${SRC_ABS}"
log "  Output tar     : ${OUTPUT_TAR}"

NO_CACHE_FLAG=()
[[ "${NO_CACHE}" == "1" ]] && NO_CACHE_FLAG=(--no-cache)

# The default "docker" buildx driver does not support file exporters
# (type=docker,dest=...).  Create a temporary docker-container builder that
# does, and clean it up on exit.
#
# env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 removes the 128KB per-step log cap so
# full compile errors appear in the build output rather than being truncated.
BUILDER_NAME="triton-local-build-$$"
docker buildx create --name "${BUILDER_NAME}" --driver docker-container \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
    --config /dev/stdin <<'BUILDKIT_CONFIG'
[registry."172.17.0.1:5005"]
  http = true
  insecure = true
BUILDKIT_CONFIG
docker buildx use "${BUILDER_NAME}"

# --build-context local_src_dir=...  satisfies COPY --from=local_src_dir in
#                                    Dockerfile.rocm_ci.
# The main build context is the ci/ directory (small; not used by the build
# itself — all source content arrives via the named context).
#
# BuildKit's --local transfer does NOT follow symlinks to directories; it sends
# them as symlink objects, leaving dangling links inside the container.
# The source tree contains at least one such symlink:
#   src/triton-inference-server-server → SERVER_REPO_PATH (likely on /home/)
#
# Fix: create a staging directory where every top-level entry is a REAL
# directory (hard-linked if possible, full copy otherwise).
#
STAGING_DIR="${WORK_DIR}/staging-$$"
mkdir -p "${STAGING_DIR}"
trap 'docker buildx rm --force "${BUILDER_NAME}" 2>/dev/null || true; rm -rf "${STAGING_DIR}"' EXIT

log "Resolving symlinks in source tree for BuildKit (staging: ${STAGING_DIR})..."
for entry in "${SRC_ABS}"/*; do
    name="$(basename "${entry}")"
    real="$(realpath "${entry}")"
    if cp -al "${real}" "${STAGING_DIR}/${name}" 2>/dev/null; then
        log "  hard-linked: ${name}"
    else
        log "  copying    : ${name} (different filesystem — may take a moment)"
        cp -r "${real}" "${STAGING_DIR}/${name}"
    fi
done
log "Staging complete."

# Use a local registry (localhost:5005) as a BuildKit cache source/destination so the
# expensive buildenv stage is not re-run from scratch on every invocation.
# The registry is started automatically if not running:
#   docker run -d --name triton-cache-registry -p 5005:5000 \
#       -v /scratch/users/diptodeb/tmp/registry-cache:/var/lib/registry registry:2
LOCAL_REGISTRY="172.17.0.1:5005"
BUILDENV_CACHE_REF="${LOCAL_REGISTRY}/triton-buildenv:${TRITON_VERSION}"
CACHE_FROM_FLAGS=()
CACHE_TO_FLAGS=()

# Check if local registry is reachable and has the buildenv image cached
if curl -sf "http://${LOCAL_REGISTRY}/v2/_catalog" >/dev/null 2>&1; then
    if curl -sf "http://${LOCAL_REGISTRY}/v2/triton-buildenv/tags/list" 2>/dev/null | grep -q "${TRITON_VERSION}"; then
        log "Local registry has cached buildenv — skipping re-build of that stage"
        CACHE_FROM_FLAGS=(--cache-from "type=registry,ref=${BUILDENV_CACHE_REF}")
    else
        log "Local registry reachable but no buildenv cache yet — will populate after build"
    fi
    # Always write the buildenv layer back to the registry so the next run is fast
    CACHE_TO_FLAGS=(--cache-to "type=registry,ref=${BUILDENV_CACHE_REF},mode=max")
else
    log "Local registry not reachable at ${LOCAL_REGISTRY} — no layer caching"
fi

if [[ "${BUILD_TARGET}" == "buildenv" ]]; then
    # Build only the buildenv stage and load into local Docker (no output tar).
    docker buildx build \
        --builder "${BUILDER_NAME}" \
        --progress=plain \
        --shm-size=16g \
        "${NO_CACHE_FLAG[@]}" \
        "${CACHE_FROM_FLAGS[@]}" \
        "${CACHE_TO_FLAGS[@]}" \
        --build-context "local_src_dir=${STAGING_DIR}" \
        --build-arg "TRITON_INFERENCE_SERVER_VERSION=${TRITON_VERSION}" \
        --build-arg "BUILD_ONNXRUNTIME=${BUILD_ONNXRUNTIME}" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --target buildenv \
        -t "${IMAGE_TAG}" \
        -f "${DOCKERFILE}" \
        --load \
        "$(dirname "${DOCKERFILE}")"
else
    docker buildx build \
        --builder "${BUILDER_NAME}" \
        --progress=plain \
        --shm-size=16g \
        "${NO_CACHE_FLAG[@]}" \
        "${CACHE_FROM_FLAGS[@]}" \
        "${CACHE_TO_FLAGS[@]}" \
        --build-context "local_src_dir=${STAGING_DIR}" \
        --build-arg "TRITON_INFERENCE_SERVER_VERSION=${TRITON_VERSION}" \
        --build-arg "BUILD_ONNXRUNTIME=${BUILD_ONNXRUNTIME}" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --target "${BUILD_TARGET}" \
        -t "${IMAGE_TAG}" \
        -f "${DOCKERFILE}" \
        --output "type=docker,dest=${OUTPUT_TAR}" \
        "$(dirname "${DOCKERFILE}")"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

banner "Build complete"
if [[ "${BUILD_TARGET}" == "buildenv" ]]; then
    log "  Image      : ${IMAGE_TAG} (loaded into local Docker)"
    log ""
    log "  To run a container and execute build.py manually (mount build dir for persistence):"
    log "    mkdir -p ${WORK_DIR}/tritonbuild"
    log "    docker run -it -v $(realpath "${WORK_DIR}")/tritonbuild:/tmp/tritonbuild ${IMAGE_TAG} bash"
    log "  Inside the container, run the build.py invocation from Dockerfile.rocm_ci (including"
    log "  --extra-core-cmake-arg BOOST_ROOT=/root/micromamba). To resume after a failure,"
    log "  re-run the container with the same volume and run ./cmake_build from /tmp/tritonbuild."
else
    log "  Output tar : ${OUTPUT_TAR}"
    if [[ -f "${OUTPUT_TAR}" ]]; then
        log "  File size  : $(du -sh "${OUTPUT_TAR}" | cut -f1)"
        log ""
        log "  To load into Docker:"
        log "    docker load -i ${OUTPUT_TAR}"
    else
        log "  WARNING: expected output not found: ${OUTPUT_TAR}"
        log "  Check the docker buildx output above for errors."
    fi
fi
log "============================================================"
