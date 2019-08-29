#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

DATADIR=/data/inferenceserver/$1

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

RET=0
rm -fr *.log

LOG_IDX=0

# LifeCycleTest.test_parse_error_noexit_strict
SERVER_ARGS="--model-repository=/tmp/xyzx --strict-readiness=true --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_noexit
SERVER_ARGS="--model-repository=/tmp/xyzx --strict-readiness=false --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_modelfail
rm -fr models
mkdir models
for i in graphdef savedmodel netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
rm models/graphdef_float32_float32_float32/*/*

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_init_error_modelfail
rm -fr models
mkdir models
cp -r ../custom_models/custom_sequence_int32 models/.
cp -r ../custom_models/custom_int32_int32_int32 models/.
sed -i "s/OUTPUT/_OUTPUT/" models/custom_sequence_int32/config.pbtxt
sed -i "s/OUTPUT/_OUTPUT/" models/custom_int32_int32_int32/config.pbtxt

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

set +e
python $LC_TEST LifeCycleTest.test_init_error_modelfail >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_model_no_version
rm -fr models
mkdir models
for i in savedmodel netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
mkdir -p models/graphdef_float32_float32_float32
cp $DATADIR/qa_model_repository/graphdef_float32_float32_float32/config.pbtxt \
    models/graphdef_float32_float32_float32/.

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

set +e
python $LC_TEST LifeCycleTest.test_parse_error_model_no_version >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_ignore_zero_prefixed_version
rm -fr models
mkdir models
for i in savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    mv models/${i}_float32_float32_float32/3 models/${i}_float32_float32_float32/003
done

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_parse_ignore_zero_prefixed_version >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# check server log for the warning messages
if [ `grep -c "ignore version directory '003' which contains leading zeros in its directory name" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_model_load_unload
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_model_load_unload_disabled
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-repository=`pwd`/models --allow-poll-model-repository=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_version_load_unload
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_version_load_unload_disabled
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --allow-poll-model-repository=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

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

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_file_delete
rm -fr models config.pbtxt.*
mkdir models
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 models/.

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 --exit-timeout-secs=5 --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_dynamic_file_delete >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_control
# enable explicit model control, no model in the repository should be loaded
rm -fr models config.pbtxt.*
mkdir models
for i in graphdef savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --allow-model-control=true \
             --allow-poll-model-repository=false --exit-timeout-secs=5 \
             --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# no model should be loaded
set +e
model_count=`curl -s localhost:8000/api/status | grep "MODEL_" | wc -l`
set -e
if [ "$model_count" != "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
python $LC_TEST LifeCycleTest.test_model_control >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1)) 

# Send HTTP request to control endpoint
rm -fr models config.pbtxt.*
mkdir models
for i in graphdef savedmodel netdef plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done

# Polling enabled (default), control API should not work
# This test also keeps using "--model-store" to ensure backward compatibility
SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=0 --exit-timeout-secs=5 --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# unload API should return bad request
set +e
code=`curl -s -w %{http_code} -X POST localhost:8000/api/modelcontrol/unload/graphdef_float32_float32_float32`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# the model should not be unloaded
set +e
unavailable_count=`curl -s localhost:8000/api/status/graphdef_float32_float32_float32 | grep "MODEL_UNAVAILABLE" | wc -l`
set -e
if [ "$unavailable_count" != "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# remove model file so that if reload is triggered, model will become unavailable
rm models/graphdef_float32_float32_float32/*/*

# load API should return bad request
set +e
code=`curl -s -w %{http_code} -X POST localhost:8000/api/modelcontrol/load/graphdef_float32_float32_float32`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# the model should still be available
set +e
unavailable_count=`curl -s localhost:8000/api/status/graphdef_float32_float32_float32 | grep "MODEL_UNAVAILABLE" | wc -l`
set -e
if [ "$unavailable_count" != "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Send HTTP request to invalid endpoints
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --allow-poll-model-repository=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
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

LOG_IDX=$((LOG_IDX+1))

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
