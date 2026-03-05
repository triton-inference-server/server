#!/bin/bash
# prefetch_sources.sh — pre-fetches all git dependencies for an air-gapped
# Triton Inference Server CI build.
#
# Usage:
#   ./prefetch_sources.sh [WORK_DIR]
#
#   WORK_DIR defaults to ./triton-ci-workspace.
#
# Environment variable overrides (all optional):
#   WORK_DIR           - workspace root (overridden by positional arg if given)
#   ROCM_TAG           - tag used for all ROCm/* repos     (default: rocm7.2_r25.12)
#   COMMON_TAG         - tag used for triton-inference-server/common (default: main)
#   SERVER_REPO_PATH   - local path to the already-cloned server repo
#                        (default: ~/devel/triton-inference-server-server)
#   BUILD_ONNXRUNTIME  - set to "ON" to also prefetch onnxruntime + AMDMIGraphX
#                        (default: ON)
#   ONNXRUNTIME_BACKEND_REPO_PATH - local path to patched onnxruntime_backend repo
#                        (if set, used instead of cloning; for in-tree ORT build)
#
# After this script completes, WORK_DIR/src/ contains:
#
#   triton-inference-server-server/       (symlink to SERVER_REPO_PATH)
#   common/
#   triton-inference-server-core/
#   triton-inference-server-backend/
#   triton-inference-server-third_party/
#   triton-inference-server-python_backend/
#   third_party/
#     curl/
#     grpc/                               (cloned with --recurse-submodules)
#     libevent/
#     nlohmann-json/
#     prometheus-cpp/
#     crc32c/
#     opentelemetry-cpp/
#
#   (when BUILD_ONNXRUNTIME=ON)
#   triton-inference-server-onnxruntime_backend/
#   onnxruntime/
#   AMDMIGraphX/
#
# NOTE on grpc submodules:
#   The git insteadOf rule in Dockerfile.triton_build redirects the top-level
#   grpc.git URL to file:///src/third_party/grpc.  grpc's own .gitmodules still
#   reference external HTTPS URLs for sub-dependencies (abseil-cpp, protobuf,
#   re2, boringssl, etc.) which are NOT covered by any insteadOf rule.
#   To support fully air-gapped builds we therefore clone grpc with
#   --recurse-submodules: all submodule trees are embedded in this local repo
#   and cmake's subsequent ExternalProject_Add/git-submodule-update step will
#   find all content present without reaching out to the internet.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORK_DIR="${1:-${WORK_DIR:-$(pwd)/triton-ci-workspace}}"
SRC_DIR="${WORK_DIR}/src"

ROCM_TAG="${ROCM_TAG:-rocm7.2_r25.12}"
COMMON_TAG="${COMMON_TAG:-main}"
SERVER_REPO_PATH="${SERVER_REPO_PATH:-${HOME}/devel/triton-inference-server-server}"
BUILD_ONNXRUNTIME="${BUILD_ONNXRUNTIME:-ON}"

ROCM_ORG="https://github.com/ROCm"
TRITON_ORG="https://github.com/triton-inference-server"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[prefetch] $*"; }
warn() { echo "[prefetch] WARNING: $*" >&2; }
die()  { echo "[prefetch] ERROR: $*" >&2; exit 1; }

# Clone a repo at a branch/tag; skip if destination already exists.
# Handles commit hashes (40-char hex) by doing a full clone + checkout.
clone_at_ref() {
    local dest="$1" url="$2" ref="$3"
    if [[ -d "${dest}/.git" ]]; then
        log "Already present, skipping: $(basename "${dest}")"
        return
    fi
    mkdir -p "$(dirname "${dest}")"
    if [[ "${ref}" =~ ^[0-9a-f]{7,40}$ ]]; then
        log "Cloning ${url} (full clone for commit ${ref})"
        git clone "${url}" "${dest}"
        git -C "${dest}" checkout "${ref}"
    else
        log "Cloning ${url} @ ${ref}"
        git clone --branch "${ref}" --single-branch "${url}" "${dest}"
    fi
}

# Clone grpc WITH all submodules.  See the NOTE at the top of this script.
clone_grpc() {
    local dest="$1" url="$2" ref="$3"
    if [[ -d "${dest}/.git" ]]; then
        log "Already present, skipping: grpc (with submodules)"
        return
    fi
    mkdir -p "$(dirname "${dest}")"
    log "Cloning grpc @ ${ref} with --recurse-submodules (may take several minutes)"
    git clone --branch "${ref}" --recurse-submodules --single-branch "${url}" "${dest}"
}

# Extract the GIT_TAG for a given repo URL fragment from a CMakeLists.txt.
# Prints the tag, or "" if not found. Uses POSIX awk (no gawk-specific match(,,array)).
extract_git_tag() {
    local cmake_file="$1"
    local repo_fragment="$2"
    awk -v pat="${repo_fragment}" '
        /GIT_REPOSITORY/ && $0 ~ pat { found=1; next }
        found && /GIT_TAG/ {
            if (match($0, /GIT_TAG[[:space:]]+/)) {
                rest = substr($0, RSTART + RLENGTH)
                if (match(rest, /"([^"]*)"/))
                    print substr(rest, RSTART + 1, RLENGTH - 2)
                else if (match(rest, /[^[:space:])"]+/))
                    print substr(rest, RSTART, RLENGTH)
                found = 0
            }
        }
        found && /GIT_REPOSITORY/ { found=0 }
    ' "${cmake_file}"
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

log "Workspace : ${WORK_DIR}"
log "src dir   : ${SRC_DIR}"
log "ROCm tag  : ${ROCM_TAG}"
log "Common tag: ${COMMON_TAG}"

mkdir -p "${SRC_DIR}/third_party"

# ---------------------------------------------------------------------------
# Step 1: Server repo (symlink — avoids re-cloning a large repo)
# ---------------------------------------------------------------------------

SERVER_LINK="${SRC_DIR}/triton-inference-server-server"
if [[ -L "${SERVER_LINK}" ]]; then
    log "Server symlink already exists: ${SERVER_LINK}"
elif [[ -d "${SERVER_LINK}" ]]; then
    log "Server directory already exists (not a symlink): ${SERVER_LINK}"
else
    [[ -d "${SERVER_REPO_PATH}" ]] \
        || die "SERVER_REPO_PATH does not exist: ${SERVER_REPO_PATH}"
    log "Symlinking server repo: ${SERVER_REPO_PATH} → ${SERVER_LINK}"
    ln -s "${SERVER_REPO_PATH}" "${SERVER_LINK}"
fi

# ---------------------------------------------------------------------------
# Step 2: Main Triton dependency repos
# ---------------------------------------------------------------------------

clone_at_ref \
    "${SRC_DIR}/common" \
    "${TRITON_ORG}/common.git" \
    "${COMMON_TAG}"

for repo in core backend third_party python_backend; do
    clone_at_ref \
        "${SRC_DIR}/triton-inference-server-${repo}" \
        "${ROCM_ORG}/triton-inference-server-${repo}.git" \
        "${ROCM_TAG}"
done

# ---------------------------------------------------------------------------
# Step 3: Extract C++ third-party versions from the third_party CMakeLists
# ---------------------------------------------------------------------------

THIRD_PARTY_CMAKE="${SRC_DIR}/triton-inference-server-third_party/CMakeLists.txt"
[[ -f "${THIRD_PARTY_CMAKE}" ]] \
    || die "Expected CMakeLists.txt not found at ${THIRD_PARTY_CMAKE}"

log "Extracting C++ third-party dependency versions from ${THIRD_PARTY_CMAKE}"

TAG_CURL=$(extract_git_tag    "${THIRD_PARTY_CMAKE}" "curl/curl")
TAG_GRPC=$(extract_git_tag    "${THIRD_PARTY_CMAKE}" "grpc/grpc")
TAG_LIBEVENT=$(extract_git_tag "${THIRD_PARTY_CMAKE}" "libevent/libevent")
TAG_JSON=$(extract_git_tag     "${THIRD_PARTY_CMAKE}" "nlohmann/json")
TAG_PROM=$(extract_git_tag     "${THIRD_PARTY_CMAKE}" "prometheus-cpp")
TAG_CRC32C=$(extract_git_tag   "${THIRD_PARTY_CMAKE}" "google/crc32c")
TAG_OTEL=$(extract_git_tag     "${THIRD_PARTY_CMAKE}" "opentelemetry-cpp")

for var in TAG_CURL TAG_GRPC TAG_LIBEVENT TAG_JSON TAG_PROM TAG_CRC32C TAG_OTEL; do
    if [[ -z "${!var}" ]]; then
        warn "Could not extract ${var} from ${THIRD_PARTY_CMAKE}; clone may fail."
        warn "You can set it manually as an env var before running this script."
    else
        log "  ${var}=${!var}"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Clone C++ third-party repos into src/third_party/
# ---------------------------------------------------------------------------

[[ -n "${TAG_CURL}" ]]    && clone_at_ref "${SRC_DIR}/third_party/curl"              "https://github.com/curl/curl.git"                              "${TAG_CURL}"
[[ -n "${TAG_LIBEVENT}" ]] && clone_at_ref "${SRC_DIR}/third_party/libevent"         "https://github.com/libevent/libevent.git"                      "${TAG_LIBEVENT}"
[[ -n "${TAG_JSON}" ]]    && clone_at_ref "${SRC_DIR}/third_party/nlohmann-json"      "https://github.com/nlohmann/json.git"                          "${TAG_JSON}"
[[ -n "${TAG_PROM}" ]]    && clone_at_ref "${SRC_DIR}/third_party/prometheus-cpp"     "https://github.com/jupp0r/prometheus-cpp.git"                  "${TAG_PROM}"
[[ -n "${TAG_CRC32C}" ]]  && clone_at_ref "${SRC_DIR}/third_party/crc32c"             "https://github.com/google/crc32c.git"                          "${TAG_CRC32C}"
[[ -n "${TAG_OTEL}" ]]    && clone_at_ref "${SRC_DIR}/third_party/opentelemetry-cpp"  "https://github.com/open-telemetry/opentelemetry-cpp.git"       "${TAG_OTEL}"

# grpc: full submodule clone (see NOTE at top of file)
[[ -n "${TAG_GRPC}" ]] && clone_grpc \
    "${SRC_DIR}/third_party/grpc" \
    "https://github.com/grpc/grpc.git" \
    "${TAG_GRPC}"

# ---------------------------------------------------------------------------
# Step 5 (optional): OnnxRuntime backend sources
# ---------------------------------------------------------------------------
# Revisions match ci/manifest.yml so prefetch and CI use the same refs.
# Override with ORT_REF and MIGX_REF if needed (e.g. ORT_REF=add_padded_batch).
ORT_REF="${ORT_REF:-add_padded_batch}"
MIGX_REF="${MIGX_REF:-concat_ai}"

if [[ "${BUILD_ONNXRUNTIME}" == "ON" ]]; then
    log "BUILD_ONNXRUNTIME=ON: prefetching onnxruntime + AMDMIGraphX"
    log "  onnxruntime ref : ${ORT_REF}"
    log "  AMDMIGraphX ref : ${MIGX_REF}"

    # Use local onnxruntime_backend repo if set (e.g. patched for in-tree ORT build)
    ORT_BACKEND_LINK="${SRC_DIR}/triton-inference-server-onnxruntime_backend"
    if [[ -n "${ONNXRUNTIME_BACKEND_REPO_PATH:-}" ]] && [[ -d "${ONNXRUNTIME_BACKEND_REPO_PATH}" ]]; then
        if [[ -d "${ORT_BACKEND_LINK}" ]] && [[ ! -L "${ORT_BACKEND_LINK}" ]]; then
            rm -rf "${ORT_BACKEND_LINK}"
        fi
        [[ ! -e "${ORT_BACKEND_LINK}" ]] && ln -s "$(realpath "${ONNXRUNTIME_BACKEND_REPO_PATH}")" "${ORT_BACKEND_LINK}"
        log "Using local onnxruntime_backend: ${ONNXRUNTIME_BACKEND_REPO_PATH}"
    else
        clone_at_ref \
            "${ORT_BACKEND_LINK}" \
            "${ROCM_ORG}/triton-inference-server-onnxruntime_backend.git" \
            "${ROCM_TAG}"
    fi

    clone_at_ref \
        "${SRC_DIR}/onnxruntime" \
        "${ROCM_ORG}/onnxruntime.git" \
        "${ORT_REF}"

    clone_at_ref \
        "${SRC_DIR}/AMDMIGraphX" \
        "${ROCM_ORG}/AMDMIGraphX.git" \
        "${MIGX_REF}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log "  Prefetch complete."
log "  Source tree:"
find "${SRC_DIR}" -maxdepth 2 -name ".git" \
    | sed "s|${SRC_DIR}/||; s|/.git||" \
    | sort \
    | while read -r p; do log "    ${p}"; done
log "============================================================"
