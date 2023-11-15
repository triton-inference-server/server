#!/bin/bash
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
UNIT_TEST=./response_cache_test
export CUDA_VISIBLE_DEVICES=0

# Only localhost supported in this test for now, but in future could make
# use of a persistent remote redis server, or similarly use --replicaof arg.
export TRITON_REDIS_HOST="localhost"
export TRITON_REDIS_PORT="6379"
REDIS_LOG="./redis-server.unit_tests.log"

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

## Unit tests currently run for both Local and Redis cache implementations
## by default. However, we could break out the unit tests for each
## into separate runs gtest filters if needed in the future:
## - `${UNIT_TEST} --gtest_filter=*Local*`
## - `${UNIT_TEST} --gtest_filter=*Redis*`
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

MODEL_DIR="${PWD}/models"
mkdir -p "${MODEL_DIR}/decoupled_cache/1"
mkdir -p "${MODEL_DIR}/identity_cache/1"

# Check that server fails to start for a "decoupled" model with cache enabled
EXTRA_ARGS="--model-control-mode=explicit --load-model=decoupled_cache"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${MODEL_DIR} --response-cache-byte-size=8192 ${EXTRA_ARGS}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Failed: $SERVER started successfully when it was expected to fail\n***"
    cat $SERVER_LOG
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
else
    # Check that server fails with the correct error message
    set +e
    grep -i "response cache does not currently support" ${SERVER_LOG} | grep -i "decoupled"
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed: Expected response cache / decoupled mode error message in output\n***"
        cat $SERVER_LOG
        RET=1
    fi
    set -e
fi

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
check_server_expected_failure "Failed to connect to Redis: Connection refused"

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

# Clean up redis server before exiting test
unset_redis_auth
stop_redis

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
