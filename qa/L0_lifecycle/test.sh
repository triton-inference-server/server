#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
LC_TEST=lifecycle_test.py

DATADIR=/data/inferenceserver

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

RET=0
rm -fr *.log

# LifeCycleTest.test_parse_error_noexit_strict
SERVER_ARGS="--model-store=/tmp/xyzx --strict-readiness=true --exit-on-error=false"
SERVER_LOG="./inference_server_0.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep 5

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit_strict >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_parse_error_noexit
SERVER_ARGS="--model-store=/tmp/xyzx --strict-readiness=false --exit-on-error=false"
SERVER_LOG="./inference_server_1.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep 5

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_parse_error_modelfail
rm -fr models
mkdir models
for i in graphdef savedmodel netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
rm models/graphdef_float32_float32_float32/*/*

SERVER_ARGS="--model-store=`pwd`/models --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_2.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

set +e
python $LC_TEST LifeCycleTest.test_parse_error_modelfail >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_dynamic_model_load_unload
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_3.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_dynamic_model_load_unload_disabled
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-store=`pwd`/models --allow-poll-model-repository=false \
             --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_4.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload_disabled >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_dynamic_version_load_unload
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_5.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_dynamic_version_load_unload_disabled
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=1 \
             --allow-poll-model-repository=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_6.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload_disabled >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# LifeCycleTest.test_dynamic_model_modify
rm -fr models config.pbtxt.*
mkdir models
for i in savedmodel plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    sed '/^version_policy/d' \
        $DATADIR/qa_model_repository/${i}_float32_float32_float32/config.pbtxt > config.pbtxt.${i}
    sed 's/output0_labels/wrong_output0_labels/' \
        $DATADIR/qa_model_repository/${i}_float32_float32_float32/config.pbtxt > config.pbtxt.wrong.${i}
    sed 's/label/label9/' \
        $DATADIR/qa_model_repository/${i}_float32_float32_float32/output0_labels.txt > \
        models/${i}_float32_float32_float32/wrong_output0_labels.txt
done

SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_7.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_modify >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Send HTTP request to invalid endpoints
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=1 \
             --allow-poll-model-repository=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_8.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} localhost:8000/noanapi/health/ready`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
code=`curl -s -w %{http_code} localhost:8000/api/notanendpoint`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# python unittest seems to swallow ImportError and still return 0 exit
# code. So need to explicitly check CLIENT_LOG to make sure we see
# some running tests
set +e
grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed To Run\n***"
    RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
