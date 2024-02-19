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

CLIENT_LOG="logging_client.log"
TEST_RESULT_FILE="test_results.txt"
LOG_TEST="logging_test.py"
SERVER_LOG="./logging_server.log"

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

MODELSDIR=${MODELDIR}/logging/models
source ../../common/util.sh

function verify_log_counts () {
  non_verbose_expected=$1
  verbose_expected=$2

  if [ `grep -c "Specific Msg!" $SERVER_LOG` != $non_verbose_expected ]; then
    echo -e "\n***\n*** Test Failed: Specific Msg Count Incorrect\n***"
    RET=1
  fi
  if [ `grep -c "Info Msg!" $SERVER_LOG` != $non_verbose_expected ]; then
    echo -e "\n***\n*** Test Failed: Info Msg Count Incorrect\n***"
    RET=1
  fi
  if [ `grep -c "Warning Msg!" $SERVER_LOG` != $non_verbose_expected ]; then
    echo -e "\n***\n*** Test Failed: Warning Msg Count Incorrect\n***"
    RET=1
  fi
  if [ `grep -c "Error Msg!" $SERVER_LOG` != $non_verbose_expected ]; then
    echo -e "\n***\n*** Test Failed: Error Msg Count Incorrect\n***"
    RET=1
  fi
  if [ `grep -c "Verbose Msg!" $SERVER_LOG` != $verbose_expected ]; then
    echo -e "\n***\n*** Test Failed: Verbose Msg Count Incorrect\n***"
    RET=1
  fi
}

rm -f *.log

# set up simple repository MODELBASE
rm -fr ${MODELSDIR} && mkdir -p ${MODELSDIR} && \
    python_model="identity_fp32_logging"
    mkdir -p models/$python_model/1/
    cp ../../python_models/${python_model}/config.pbtxt models/${python_model}/config.pbtxt
    cp ../../python_models/${python_model}/model.py models/${python_model}/1/
RET=0

#Run Server with Default Log Settings
SERVER_ARGS="--model-repository=${MODELSDIR} --backend-directory=${BACKEND_DIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SUBTEST="default"
python3 -m pytest --junitxml=log_test.${SUBTEST}.report.xml ${LOG_TEST} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill_server

# Check if correct # log messages are present [ non-verbose-msg-cnt | verbose-msg-cnt ]
# NOTE: Windows does not seem to have a way to send a true SIGINT signal
# to tritonserver. Instead, it seems required to use taskkill.exe with /F (force)
# to kill the running program. This means the server terminates immediately,
# instead of shutting down how it would if Ctrl^C was invoked from the terminal.
# To properly test functionality, we need a WAR. In the meantime, we will subtract
# 1 from the expected values to account for the fact that no logs will be emitted
# from the finalize function.
if [[ ${TEST_WINDOWS} == 1 ]]; then
    verify_log_counts 3 0
else
    verify_log_counts 4 0
fi


rm -f *.log
#Run Server Enabling Verbose Messages
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Enable verbose logging
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":1}' ${TRITONSERVER_IPADDR}:8000/v2/logging`

if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** Test Failed: Could not Change Log Settings\n***"
    RET=1
fi

SUBTEST="verbose"
python3 -m pytest --junitxml=log_test.${SUBTEST}.report.xml ${LOG_TEST} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill_server

# Verbose only 3 because model must initialize before
# log settings can be modified
if [[ ${TEST_WINDOWS} == 1 ]]; then
    verify_log_counts 3 2
else
    verify_log_counts 4 3
fi

rm -f *.log
#Run Server Enabling Verbose Messages
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Disable all logging
BOOL_PARAMS=${BOOL_PARAMS:="log_info log_warning log_error"}
for BOOL_PARAM in $BOOL_PARAMS; do
    # Attempt to use integer instead of bool
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":false}' ${TRITONSERVER_IPADDR}:8000/v2/logging`
    if [ "$code" != "200" ]; then
        cat ./curl.out
        echo -e "\n***\n*** Test Failed: Could not Change Log Settings\n***"
        RET=1
    fi
done

SUBTEST="disabled"
python3 -m pytest --junitxml=log_test.${SUBTEST}.report.xml ${LOG_TEST} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill_server

# Will have 1 occurrence of each non-verbose log type
# because the server must initialize before log settings
# can be modified
# Same count for both Unix and Windows because this does
# not test log output in the finalize step.
verify_log_counts 1 0


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Logging test PASSED. \n***"
else
    echo -e "\n***\n*** Logging test FAILED. \n***"
fi

exit $RET
