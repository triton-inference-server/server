#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=20
source ../common/util.sh


SERVER_LOG_BASE="./inference_server"
rm -f $SERVER_LOG_BASE*

NEGATIVE_TEST_ARGS="--model-repository=`pwd`/models --default-max-batch-size=-1 --strict-model-config=false --log-verbose=1"

POSITIVE_TEST_ARGS=("--model-repository=`pwd`/models --default-max-batch-size=3 --strict-model-config=false --log-verbose=1" \
                    "--model-repository=`pwd`/models --default-max-batch-size=3 --backend-config=tensorflow,default-max-batch-size=5 --strict-model-config=false --log-verbose=1" \
                    "--model-repository=`pwd`/models --backend-config=tensorflow,default-max-batch-size=5 --strict-model-config=false --log-verbose=1" \
                    "--model-repository=`pwd`/models --strict-model-config=false --log-verbose=1")

# These integers correspond to the expected default-max-batch-size which gets set 
# in the POSITIVE_TEST_ARGS
POSITIVE_TEST_ANSWERS=(3 5 5 4)

RET=0
# Positive tests
for ((i=0; i < ${#POSITIVE_TEST_ARGS[@]}; i++)); do
    SERVER_ARGS=${POSITIVE_TEST_ARGS[$i]}
    SERVER_LOG=$SERVER_LOG_BASE.backend_config_positive_$i.log
    run_server
    
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "*** FAILED: Server failed to start $SERVER\n"
        RET=1

    else
        RESULT_LOG_LINE=$(grep "Resolved default max batch size to:" $SERVER_LOG)
        if [ "$RESULT_LOG_LINE" != "" ]; then
            
            # Pick out the logged value of the default-max-batch-size which gets passed into model creation
            RESOLVED_DEFAULT_MAX_BATCH_SIZE=$(awk -v line="$RESULT_LOG_LINE" 'BEGIN {split(line, a, "]"); split(a[2], b, ": "); print b[2]}')

            if [ "$RESOLVED_DEFAULT_MAX_BATCH_SIZE" != "${POSITIVE_TEST_ANSWERS[$i]}" ]; then
                echo "*** FAILED: Found default-max-batch-size not equal to the expected default-max-batch-size. Expected: ${POSITIVE_TEST_ANSWERS[$i]}, Found: $RESOLVED_DEFAULT_MAX_BATCH_SIZE \n" 
                RET=1
            fi
        else
            echo "*** FAILED: No log statement stating default amx batch size\n"
            RET=1
        fi
        
        kill $SERVER_PID
        wait $SERVER_PID
    fi

done

# Negative test
SERVER_ARGS=$NEGATIVE_TEST_ARGS
SERVER_LOG=$SERVER_LOG_BASE.backend_config_negative.log
run_server

if [ "$SERVER_PID" == "0" ]; then
    if ! grep "setting default max batch size: Invalid argument" $SERVER_LOG; then
        echo -e "*** FAILED: Expected invalid default max batch size message but found other error.\n"
        RET=1
    fi
else
    echo -e "*** FAILED: Expected server to exit with error, but found running.\n"
    RET=1
    kill $SERVER_PID
    wait $SERVER_PID
fi

# Print test outcome
if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

