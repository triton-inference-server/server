#!/bin/bash
# Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

RET=0

TEST_LOG="./response_cache_test.log"
UNIT_TEST="./response_cache_test --gtest_output=xml:response_cache.report.xml"
export CUDA_VISIBLE_DEVICES=0

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "${REPO_VERSION}" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi
# Only localhost supported in this test for now, but in future could make
# use of a persistent remote redis server, or similarly use --replicaof arg.
export TRITON_REDIS_HOST="localhost"
export TRITON_REDIS_PORT="6379"
REDIS_LOG="./redis-server.unit_tests.log"
ENSEMBLE_CACHE_TEST_PY="./ensemble_cache_test.py"
SERVER=/opt/tritonserver/bin/tritonserver
CLIENT_LOG="./client.log"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG=./inference_server.log
RESET_CONFIG_FUNCTION="_reset_config_files"
CACHE_SIZE=10840
source ../common/util.sh

MODEL_DIR="${PWD}/models"
ENSEMBLE_MODEL_DIR="${MODEL_DIR}/ensemble_models"
ENSEMBLE_CACHE_DECOUPLED="${MODEL_DIR}/ensemble_cache_decoupled"
ENSEMBLE_CACHE_COMPOSING_DECOUPLED="${MODEL_DIR}/ensemble_cache_composing_decoupled"
rm -fr ${ENSEMBLE_MODEL_DIR} && mkdir ${ENSEMBLE_MODEL_DIR}
rm -fr ${ENSEMBLE_CACHE_DECOUPLED} && mkdir ${ENSEMBLE_CACHE_DECOUPLED}
rm -fr ${ENSEMBLE_CACHE_COMPOSING_DECOUPLED} && mkdir ${ENSEMBLE_CACHE_COMPOSING_DECOUPLED}
ENSEMBLE_MODEL="simple_graphdef_float32_float32_float32"
COMPOSING_MODEL="graphdef_float32_float32_float32"

cp -r "/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository/${ENSEMBLE_MODEL}" "${ENSEMBLE_MODEL_DIR}/${ENSEMBLE_MODEL}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_model_repository/${COMPOSING_MODEL}" "${ENSEMBLE_MODEL_DIR}/${COMPOSING_MODEL}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository/${ENSEMBLE_MODEL}" "${ENSEMBLE_CACHE_DECOUPLED}/${ENSEMBLE_MODEL}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_model_repository/${COMPOSING_MODEL}" "${ENSEMBLE_CACHE_DECOUPLED}/${COMPOSING_MODEL}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository/${ENSEMBLE_MODEL}" "${ENSEMBLE_CACHE_COMPOSING_DECOUPLED}/${ENSEMBLE_MODEL}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_model_repository/${COMPOSING_MODEL}" "${ENSEMBLE_CACHE_COMPOSING_DECOUPLED}/${COMPOSING_MODEL}"
mkdir -p "${MODEL_DIR}/decoupled_cache/1"
mkdir -p "${MODEL_DIR}/identity_cache/1"

echo -e "response_cache { enable: True }" >> "${ENSEMBLE_CACHE_DECOUPLED}/${ENSEMBLE_MODEL}/config.pbtxt"
echo -e "model_transaction_policy { decoupled: True }" >> "${ENSEMBLE_CACHE_DECOUPLED}/${ENSEMBLE_MODEL}/config.pbtxt"
echo -e "response_cache { enable: True }" >> "${ENSEMBLE_CACHE_COMPOSING_DECOUPLED}/${ENSEMBLE_MODEL}/config.pbtxt"
echo -e "model_transaction_policy { decoupled: True }" >> "${ENSEMBLE_CACHE_COMPOSING_DECOUPLED}/${COMPOSING_MODEL}/config.pbtxt"

rm -fr *.log

function install_redis() {
  ## Install redis if not already installed
  if ! command -v redis-server >/dev/null 2>&1; then
    apt update -y && apt install -y redis
  fi
}

function start_redis() {
  # Run redis server in background
  redis-server                    \
    --daemonize yes               \
    --port "${TRITON_REDIS_PORT}" \
    --logfile "${REDIS_LOG}"      \
    --loglevel debug

  # Check redis server is running
  REDIS_PING_RESPONSE=$(redis-cli -h ${TRITON_REDIS_HOST} -p ${TRITON_REDIS_PORT} ping)
  if [ "${REDIS_PING_RESPONSE}" == "PONG" ]; then
    echo "Redis successfully started in background"
  else
    echo -e "\n***\n*** Failed: Redis server did not start successfully\n***"
    RET=1
  fi
}

function stop_redis() {
  echo "Stopping Redis server..."
  redis-cli -h "${TRITON_REDIS_HOST}" -p "${TRITON_REDIS_PORT}" shutdown || true
  echo "Redis server shutdown"
}

function set_redis_auth() {
  # NOTE: Per-user auth [Access Control List (ACL)] is only supported in
  #       Redis >= 6.0 and is more comprehensive in what can be configured.
  #       For simplicity and wider range of Redis version support, use
  #       server-wide password  via "requirepass" for now.
  redis-cli -h "${TRITON_REDIS_HOST}" -p "${TRITON_REDIS_PORT}" config set requirepass "${REDIS_PW}"
  export REDISCLI_AUTH="${REDIS_PW}"
}

function unset_redis_auth() {
  # Authenticate implicitly via REDISCLI_AUTH env var, then unset password/var
  redis-cli -h "${TRITON_REDIS_HOST}" -p "${TRITON_REDIS_PORT}" config set requirepass ""
  unset REDISCLI_AUTH
}

# UNIT TESTS
set +e

# Unit tests currently run for both Local and Redis cache implementations
# by default. However, we could break out the unit tests for each
# into separate runs gtest filters if needed in the future:
# - `${UNIT_TEST} --gtest_filter=*Local*`
# - `${UNIT_TEST} --gtest_filter=*Redis*`
install_redis
# Stop any existing redis server first for good measure
stop_redis
start_redis
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH $UNIT_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** Response Cache Unit Test Failed\n***"
    RET=1
fi
stop_redis
set -e

# SERVER TESTS
function check_server_success_and_kill {
    if [ "${SERVER_PID}" == "0" ]; then
        echo -e "\n***\n*** Failed to start ${SERVER}\n***"
        cat ${SERVER_LOG}
        RET=1
    else
        kill ${SERVER_PID}
        wait ${SERVER_PID}
    fi
}

function check_server_expected_failure {
    EXPECTED_MESSAGE="${1}"
    if [ "${SERVER_PID}" != "0" ]; then
        echo -e "\n***\n*** Failed: ${SERVER} started successfully when it was expected to fail\n***"
        cat ${SERVER_LOG}
        RET=1

        kill ${SERVER_PID}
        wait ${SERVER_PID}
    else
        # Check that server fails with the correct error message
        set +e
        grep -i "${EXPECTED_MESSAGE}" ${SERVER_LOG}
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Failed: Expected [${EXPECTED_MESSAGE}] error message in output\n***"
            cat $SERVER_LOG
            RET=1
        fi
        set -e
    fi
}

# DECOUPLED MODEL TESTS
function check_server_failure_decoupled_model {
  MODEL_REPOSITORY="${1}"
  MODEL="${2}"
  EXTRA_ARGS="--model-control-mode=explicit --load-model=${MODEL}"
  SERVER_ARGS="--model-repository=${MODEL_REPOSITORY} --cache-config local,size=10480 ${EXTRA_ARGS}"

  rm -f ${SERVER_LOG}
  run_server
  if [ "${SERVER_PID}" != "0" ]; then
    echo -e "\n***\n*** Failed: ${SERVER} started successfully when it was expected to fail\n***"
    cat ${SERVER_LOG}
    RET=1

    kill ${SERVER_PID}
    wait ${SERVER_PID}
  else
    # Check that server fails with the correct error message
    set +e
    grep -i "response cache does not currently support" ${SERVER_LOG} | grep -i "decoupled"
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed: Expected response cache / decoupled mode error message in output\n***"
        cat ${SERVER_LOG}
        RET=1
    fi
    set -e
  fi
}

# ENSEMBLE CACHE TESTS
function test_response_cache_ensemble_model {
  TESTCASE="${1}"
  ERROR_MESSAGE="${2}"
  SERVER_ARGS="--model-repository=${ENSEMBLE_MODEL_DIR} --cache-config local,size=${CACHE_SIZE} --model-control-mode=explicit"
  run_server
  set +e
  python ${ENSEMBLE_CACHE_TEST_PY} ${TESTCASE} >> ${CLIENT_LOG} 2>&1
  if [ $? -ne 0 ]; then
      RET=1
  else
      check_test_results ${TEST_RESULT_FILE} 1
      if [ $? -ne 0 ]; then
          cat ${CLIENT_LOG}
          echo -e ${ERROR_MESSAGE}
          RET=1
      fi
  fi

  if [ "${TESTCASE}" = "EnsembleCacheTest.test_ensemble_cache_insertion_failure" ]; then
      # Check for the error message in the log file
      set +e
      grep -i "Failed to insert key" "${SERVER_LOG}"
      if [ $? -ne 0 ]; then
          echo "\n***\n*** Failed: Cache insertion successful when it was expected to fail\n***"
          RET=1
      fi
      set -e
  fi
  set -e
  check_server_success_and_kill
}

# Check that server fails to start for a "decoupled" model with cache enabled
check_server_failure_decoupled_model ${MODEL_DIR}  "decoupled_cache"

# Test with model expected to load successfully
EXTRA_ARGS="--model-control-mode=explicit --load-model=identity_cache"

# Test old cache config method
# --response-cache-byte-size must be non-zero to test models with cache enabled
SERVER_ARGS="--model-repository=${MODEL_DIR} --response-cache-byte-size=8192 ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test new cache config method
SERVER_ARGS="--model-repository=${MODEL_DIR} --cache-config=local,size=8192 ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test that specifying multiple cache types is not supported and should fail
SERVER_ARGS="--model-repository=${MODEL_DIR} --cache-config=local,size=8192 --cache-config=redis,key=value ${EXTRA_ARGS}"
run_server
check_server_expected_failure "multiple cache configurations"

# Test that specifying both config styles is incompatible and should fail
SERVER_ARGS="--model-repository=${MODEL_DIR} --response-cache-byte-size=12345 --cache-config=local,size=67890 ${EXTRA_ARGS}"
run_server
check_server_expected_failure "incompatible flags"

## Redis Cache CLI tests
REDIS_ENDPOINT="--cache-config redis,host=${TRITON_REDIS_HOST} --cache-config redis,port=${TRITON_REDIS_PORT}"
REDIS_LOG="./redis-server.cli_tests.log"
start_redis

# Test simple redis cache config succeeds
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test triton fails to initialize if it can't connect to redis cache
SERVER_ARGS="--model-repository=${MODEL_DIR} --cache-config=redis,host=localhost --cache-config=redis,port=nonexistent ${EXTRA_ARGS}"
run_server
check_server_expected_failure "failed to connect to Redis (localhost:0): Connection refused"

# Test triton fails to initialize if it can't resolve host for redis cache
SERVER_ARGS="--model-repository=${MODEL_DIR} --cache-config=redis,host=nonexistent --cache-config=redis,port=nonexistent ${EXTRA_ARGS}"
run_server
# Either of these errors can be returned for bad hostname, so check for either.
MSG1="Temporary failure in name resolution"
MSG2="Name or service not known"
check_server_expected_failure "${MSG1}\|${MSG2}"

# Test triton fails to initialize if minimum required args (host & port) not all provided
SERVER_ARGS="--model-repository=${MODEL_DIR} --cache-config=redis,port=${TRITON_REDIS_HOST} ${EXTRA_ARGS}"
run_server
check_server_expected_failure "Must at a minimum specify"

## Redis Authentication tests

# Automatically provide auth via REDISCLI_AUTH env var when set: https://redis.io/docs/ui/cli/
REDIS_PW="redis123!"
set_redis_auth

### Credentials via command-line

# Test simple redis authentication succeeds with correct credentials
REDIS_CACHE_AUTH="--cache-config redis,password=${REDIS_PW}"
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${REDIS_CACHE_AUTH} ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test simple redis authentication fails with wrong credentials
REDIS_CACHE_AUTH="--cache-config redis,password=wrong"
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${REDIS_CACHE_AUTH} ${EXTRA_ARGS}"
run_server
check_server_expected_failure "WRONGPASS"

# Test simple redis authentication fails with no credentials
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${EXTRA_ARGS}"
run_server
check_server_expected_failure "NOAUTH Authentication required"

### Credentials via environment variables

# Test simple redis authentication succeeds with password-only via env vars
# No username means use "default" as the username
unset TRITONCACHE_REDIS_USERNAME
export TRITONCACHE_REDIS_PASSWORD="${REDIS_PW}"
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test simple redis authentication succeeds with correct user and password via env vars
export TRITONCACHE_REDIS_USERNAME="default"
export TRITONCACHE_REDIS_PASSWORD="${REDIS_PW}"
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${EXTRA_ARGS}"
run_server
check_server_success_and_kill

# Test simple redis authentication fails with wrong credentials via env vars
export TRITONCACHE_REDIS_PASSWORD="wrong"
SERVER_ARGS="--model-repository=${MODEL_DIR} ${REDIS_ENDPOINT} ${EXTRA_ARGS}"
run_server
check_server_expected_failure "WRONGPASS"
unset TRITONCACHE_REDIS_USERNAME
unset TRITONCACHE_REDIS_PASSWORD
# Clean up redis server
unset_redis_auth
stop_redis

# Test ensemble model with cache and decoupled mode enabled
check_server_failure_decoupled_model ${ENSEMBLE_CACHE_DECOUPLED} ${ENSEMBLE_MODEL}

# Test ensemble model with cache enabled and decoupled mode enabled in composing model
check_server_failure_decoupled_model ${ENSEMBLE_CACHE_COMPOSING_DECOUPLED} ${ENSEMBLE_MODEL}

# Test ensemble model with response cache enabled
TEST_NAME="EnsembleCacheTest.test_ensemble_top_level_response_cache"
ERROR_MESSAGE="\n***\n*** Failed: Expected top level response caching\n***"
test_response_cache_ensemble_model "${TEST_NAME}" "${ERROR_MESSAGE}"

# Test ensemble model with cache enabled in all models
TEST_NAME="EnsembleCacheTest.test_ensemble_all_models_cache_enabled"
ERROR_MESSAGE="\n***\n*** Failed: Expected cache to return top-level request's response\n***"
test_response_cache_ensemble_model "${TEST_NAME}" "${ERROR_MESSAGE}"

# Test composing model cache enabled
TEST_NAME="EnsembleCacheTest.test_ensemble_composing_model_cache_enabled"
ERROR_MESSAGE="\n***\n*** Failed: Expected only composing model's input/output to be inserted in cache\n***"
test_response_cache_ensemble_model "${TEST_NAME}" "${ERROR_MESSAGE}"

# Test cache insertion failure
TEST_NAME="EnsembleCacheTest.test_ensemble_cache_insertion_failure"
ERROR_MESSAGE="\n***\n*** Failed: Request added to cache successfully when it was expected to fail\n***"
CACHE_SIZE=200
test_response_cache_ensemble_model "${TEST_NAME}" "${ERROR_MESSAGE}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
