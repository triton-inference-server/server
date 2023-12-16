#!/bin/bash
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}
LC_TEST=lifecycle_test.py
SLEEP_TIME=10
SERVER=/opt/tritonserver/bin/tritonserver
TEST_RESULT_FILE='test_results.txt'
source ../common/util.sh

function check_unit_test() {
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
}

RET=0
rm -fr *.log

LOG_IDX=0

if [ `ps | grep -c "tritonserver"` != "0" ]; then
    echo -e "Tritonserver already running"
    echo -e `ps | grep tritonserver`
    exit 1
fi

# LifeCycleTest.test_parse_error_noexit_strict
SERVER_ARGS="--model-repository=/tmp/xyzx --strict-readiness=true \
             --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep $SLEEP_TIME

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_noexit
SERVER_ARGS="--model-repository=/tmp/xyzx --strict-readiness=false \
             --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep $SLEEP_TIME


rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_noexit_strict (multiple model repositories)
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=/tmp/xyzx --model-repository=`pwd`/models \
             --strict-readiness=true --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep $SLEEP_TIME


rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_noexit (multiple model repositories)
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models --model-repository=/tmp/xyzx \
             --strict-readiness=false --exit-on-error=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_nowait
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
sleep $SLEEP_TIME


rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# GRPC Port Collision Test
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./stub_inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
SAVED_SERVER_PID=$SERVER_PID
SERVER_ARGS="--model-repository=`pwd`/models --http-port 8003 --metrics-port 8004"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
sleep $SLEEP_TIME
# check server log for the warning messages
if [ `grep -c "failed to start GRPC service: Unavailable - Socket '0.0.0.0:8001' already in use" $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not report GRPC port collision\n***"
    echo -e "\n***\n*** Test Failed\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
fi

SERVER_PID=$SAVED_SERVER_PID
kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# HTTP Port Collision Test
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./stub_inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
SAVED_SERVER_PID=$SERVER_PID
SERVER_ARGS="--model-repository=`pwd`/models --grpc-port 8003 --metrics-port 8004"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
sleep $SLEEP_TIME
# check server log for the warning messages
if [ `grep -c "failed to start HTTP service: Unavailable - Socket '0.0.0.0:8000' already in use" $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not report HTTP port collision\n***"
    echo -e "\n***\n*** Test Failed\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
fi

SERVER_PID=$SAVED_SERVER_PID

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Metrics Port Collision Test
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./stub_inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
SAVED_SERVER_PID=$SERVER_PID
SERVER_ARGS="--model-repository=`pwd`/models --grpc-port 8003 --http-port 8004"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
sleep $SLEEP_TIME
# check server log for the warning messages
if [ `grep -c "failed to start Metrics service: Unavailable - Socket '0.0.0.0:8002' already in use" $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not report metrics port collision\n***"
    echo -e "\n***\n*** Test Failed\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
fi

SERVER_PID=$SAVED_SERVER_PID

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Multiple Port Collisions Test
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
SAVED_SERVER_PID=$SERVER_PID
run_server
sleep $SLEEP_TIME
# check server log for the warning messages
if [ `grep -c "failed to start.*service: Unavailable - Socket '.*' already in use" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not report port collision\n***"
    echo -e "\n***\n*** Test Failed\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
fi

SERVER_PID=$SAVED_SERVER_PID

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# No Port Collision Test
rm -rf models
mkdir models
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

LOG_IDX=$((LOG_IDX+1))
SERVER_LOG="./inference_server_$LOG_IDX.log"

SAVED_SERVER_PID=$SERVER_PID
SERVER_ARGS="--model-repository=`pwd`/models --grpc-port 8003 --http-port 8004 --metrics-port 8005"
run_server
sleep $SLEEP_TIME
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID
kill $SAVED_SERVER_PID
wait $SAVED_SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_modelfail
rm -fr models models_0
mkdir models models_0
for i in graphdef savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
for i in onnx plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done
# Change the model files so that multiple versions will be loaded, and one of
# the versions will fail to load and cause all other versions to be unloaded.
rm models/graphdef_float32_float32_float32/3/*

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_modelfail >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_modelfail_nostrict
SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --exit-on-error=false --exit-timeout-secs=5 --strict-readiness=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_modelfail_nostrict >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_no_model_config
rm -fr models models_0
mkdir models models_0
for i in graphdef savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
for i in onnx plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done
rm models/graphdef_float32_float32_float32/config.pbtxt

# Autocomplete should not be turned on for this test because it asserts an error was logged
# when in strict model configuration mode.
SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --exit-on-error=false --exit-timeout-secs=5 --strict-model-config=true"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_no_model_config >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

# check server log for the warning messages
if [ `grep -c "failed to open text file for read" $SERVER_LOG` == "0" ] || [ `grep -c "graphdef_float32_float32_float32/config.pbtxt: No such file or directory" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not print model load failure\n***"
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_init_error_modelfail
rm -fr models models_0
mkdir models models_0
cp -r $DATADIR/qa_sequence_model_repository/onnx_sequence_int32 models/.
cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 models_0/.
sed -i "s/OUTPUT/_OUTPUT/" models/onnx_sequence_int32/config.pbtxt
sed -i "s/OUTPUT/_OUTPUT/" models_0/onnx_int32_int32_int32/config.pbtxt
for i in graphdef savedmodel; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --exit-on-error=false --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_init_error_modelfail >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_error_model_no_version
rm -fr models
mkdir models
for i in savedmodel onnx plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
mkdir -p models/graphdef_float32_float32_float32
cp $DATADIR/qa_model_repository/graphdef_float32_float32_float32/config.pbtxt \
    models/graphdef_float32_float32_float32/.

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server_tolive
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# give plenty of time for model to load (and fail to load)
wait_for_model_stable $SERVER_TIMEOUT

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_model_no_version >>$CLIENT_LOG 2>&1
check_unit_test
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

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_ignore_zero_prefixed_version >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

# check server log for the warning messages
if [ `grep -c "ignore version directory '003' which contains leading zeros in its directory name" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_parse_ignore_non_intergral_version
rm -fr models
mkdir models
for i in savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    mv models/${i}_float32_float32_float32/3 models/${i}_float32_float32_float32/abc
done

SERVER_ARGS="--model-repository=`pwd`/models --exit-on-error=false \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_ignore_non_intergral_version >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

# check server log for the warning messages
if [ `grep -c "ignore version directory 'abc' which fails to convert to integral number" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_model_load_unload
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef onnx plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 \
             --model-control-mode=poll --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_model_load_unload_disabled
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef onnx plan; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=none \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload_disabled >>$CLIENT_LOG 2>&1
check_unit_test
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

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 \
             --model-control-mode=poll --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload >>$CLIENT_LOG 2>&1
check_unit_test
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

# Show model control mode will override deprecated model control options
SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=none \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload_disabled >>$CLIENT_LOG 2>&1
check_unit_test
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

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 \
             --model-control-mode=poll --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_modify >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_dynamic_file_delete
rm -fr models config.pbtxt.*
mkdir models
for i in savedmodel plan; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --repository-poll-secs=1 \
             --model-control-mode=poll --exit-timeout-secs=5 --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_dynamic_file_delete >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_multiple_model_repository_polling
rm -fr models models_0 savedmodel_float32_float32_float32
mkdir models models_0
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 models/. && \
    rm -rf models/savedmodel_float32_float32_float32/3

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=poll --repository-poll-secs=1 --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_polling >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_multiple_model_repository_control
rm -fr models models_0 savedmodel_float32_float32_float32
mkdir models models_0
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 .
cp -r $DATADIR/qa_model_repository/savedmodel_float32_float32_float32 models/. && \
    rm -rf models/savedmodel_float32_float32_float32/3

# Show model control mode will override deprecated model control options
SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=explicit \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_control >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_control
rm -fr models config.pbtxt.*
mkdir models
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models/.
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/simple_${i}_float32_float32_float32/config.pbtxt
done

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
             --exit-timeout-secs=5 --strict-model-config=false
             --strict-readiness=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_model_control >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_control_fail
rm -fr models config.pbtxt.*
mkdir models
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    # Remove all model files so the model will fail to load
    rm models/${i}_float32_float32_float32/*/*
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
done

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
             --exit-timeout-secs=5 --strict-model-config=false
             --strict-readiness=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_model_control_fail >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_control_ensemble
rm -fr models config.pbtxt.*
mkdir models
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models/.
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/simple_${i}_float32_float32_float32/config.pbtxt
done

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
             --exit-timeout-secs=5 --strict-model-config=false
             --strict-readiness=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_model_control_ensemble >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_multiple_model_repository_control_startup_models
rm -fr models models_0 config.pbtxt.*
mkdir models models_0
# Ensemble models in the second repository
for i in plan onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models_0/.
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models_0/simple_${i}_float32_float32_float32/config.pbtxt
done

# savedmodel doesn't load because it is duplicated in 2 repositories
for i in savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=explicit \
             --strict-readiness=false \
             --strict-model-config=false --exit-on-error=false \
             --load-model=savedmodel_float32_float32_float32 \
             --load-model=plan_float32_float32_float32 \
             --load-model=simple_onnx_float32_float32_float32"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_control_startup_models >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Test loading all models on startup in EXPLICIT model control mode, re-use
# existing LifeCycleTest.test_multiple_model_repository_control_startup_models
# unit test
rm -fr models models_0 config.pbtxt.*
mkdir models models_0
# Ensemble models in the second repository
for i in plan onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models_0/.
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models_0/simple_${i}_float32_float32_float32/config.pbtxt
done

# savedmodel doesn't load because it is duplicated in 2 repositories
for i in savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=explicit \
             --strict-readiness=false \
             --strict-model-config=false --exit-on-error=false \
             --load-model=*"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_control_startup_models >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Test loading all models on startup in EXPLICIT model control mode AND
# an additional --load-model argument, it should fail
rm -fr models
mkdir models
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    sed -i "s/max_batch_size:.*/max_batch_size: 1/" models/${i}_float32_float32_float32/config.pbtxt
done

# --load-model=* can not be used with any other --load-model arguments
# as it's unclear what the user's intentions are.
SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=explicit \
             --strict-readiness=true \
             --exit-on-error=true \
             --load-model=* \
             --load-model=onnx_float32_float32_float32"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Failed: $SERVER started successfully when it was expected to fail\n***"
    cat $SERVER_LOG
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi

LOG_IDX=$((LOG_IDX+1))

# Test loading a startup model that doesn't exist, it should fail
rm -fr models && mkdir models
INVALID_MODEL="does-not-exist"
SERVER_ARGS="--model-repository=`pwd`/models \
             --model-control-mode=explicit \
             --strict-readiness=true \
             --exit-on-error=true \
             --load-model=${INVALID_MODEL}"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Failed: $SERVER started successfully when it was expected to fail\n***"
    echo -e "ERROR: Startup model [${INVALID_MODEL}] should have failed to load."
    cat $SERVER_LOG
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
# check server log for the error messages to make sure they're printed
if [ `grep -c "model not found in any model repository" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not print model load failure for non-existent model\n***"
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_repository_index
rm -fr models models_0 config.pbtxt.*
mkdir models models_0
# Ensemble models in the second repository
for i in graphdef savedmodel ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_ensemble_model_repository/qa_model_repository/simple_${i}_float32_float32_float32 models_0/.
done

# onnx doesn't load because it is duplicated in 2 repositories
for i in onnx ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models_0/.
done

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models_0 \
             --model-control-mode=explicit \
             --strict-readiness=false \
             --strict-model-config=false --exit-on-error=false \
             --load-model=onnx_float32_float32_float32 \
             --load-model=graphdef_float32_float32_float32 \
             --load-model=simple_savedmodel_float32_float32_float32"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_model_repository_index >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_model_availability_on_reload
for protocol in grpc http; do
    if [[ $protocol == "grpc" ]]; then
       export TRITONSERVER_USE_GRPC=1
    fi
    rm -fr models config.pbtxt.*
    mkdir models
    cp -r identity_zero_1_int32 models/. && mkdir -p models/identity_zero_1_int32/1

    SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
                 --exit-timeout-secs=5 --strict-model-config=false \
                 --load-model=identity_zero_1_int32 \
                 --strict-readiness=false"
    SERVER_LOG="./inference_server_$LOG_IDX.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    rm -f $CLIENT_LOG
    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    unset TRITONSERVER_USE_GRPC

    LOG_IDX=$((LOG_IDX+1))
done

# LifeCycleTest.test_model_availability_on_reload_2
for protocol in grpc http; do
    if [[ $protocol == "grpc" ]]; then
       export TRITONSERVER_USE_GRPC=1
    fi
    rm -fr models config.pbtxt.*
    mkdir models
    cp -r identity_zero_1_int32 models/. \
        && mkdir -p models/identity_zero_1_int32/1 \
        && mkdir -p models/identity_zero_1_int32/2
    echo "version_policy: { specific { versions: [1] }}" >> models/identity_zero_1_int32/config.pbtxt
    cp identity_zero_1_int32/config.pbtxt config.pbtxt.v2
    echo "version_policy: { specific { versions: [2] }}" >> config.pbtxt.v2

    SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
                 --exit-timeout-secs=5 --strict-model-config=false \
                 --load-model=identity_zero_1_int32 \
                 --strict-readiness=false"
    SERVER_LOG="./inference_server_$LOG_IDX.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    rm -f $CLIENT_LOG
    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload_2 >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
            cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    unset TRITONSERVER_USE_GRPC

    LOG_IDX=$((LOG_IDX+1))
done

# LifeCycleTest.test_model_availability_on_reload_3
for protocol in grpc http; do
    if [[ $protocol == "grpc" ]]; then
       export TRITONSERVER_USE_GRPC=1
    fi
    rm -fr models config.pbtxt.*
    mkdir models
    cp -r identity_zero_1_int32 models/. \
        && mkdir -p models/identity_zero_1_int32/1 \
        && mkdir -p models/identity_zero_1_int32/2
    echo "version_policy: { specific { versions: [1] }}" >> models/identity_zero_1_int32/config.pbtxt
    cp models/identity_zero_1_int32/config.pbtxt config.pbtxt.new

    SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
                 --exit-timeout-secs=5 --strict-model-config=false \
                 --load-model=identity_zero_1_int32 \
                 --strict-readiness=false"
    SERVER_LOG="./inference_server_$LOG_IDX.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    rm -f $CLIENT_LOG
    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload_3 >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    unset TRITONSERVER_USE_GRPC

    LOG_IDX=$((LOG_IDX+1))
done

# LifeCycleTest.test_model_reload_fail
rm -fr models config.pbtxt.*
mkdir models
cp -r identity_zero_1_int32 models/. && \
    mkdir -p models/identity_zero_1_int32/1 && \
    cp libtriton_identity.so models/identity_zero_1_int32/1/. && \
    mkdir -p models/identity_zero_1_int32/2 && \
    cp libtriton_identity.so models/identity_zero_1_int32/2/.
echo "version_policy: { specific { versions: [1] }}" >> models/identity_zero_1_int32/config.pbtxt
cp identity_zero_1_int32/config.pbtxt config.pbtxt.v2.gpu && \
    echo "version_policy: { specific { versions: [2] }}" >> config.pbtxt.v2.gpu && \
    sed -i "s/KIND_CPU/KIND_GPU/" config.pbtxt.v2.gpu

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
             --exit-timeout-secs=5 --strict-model-config=false \
             --load-model=identity_zero_1_int32 \
             --strict-readiness=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_model_reload_fail >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

# check server log for the warning messages
if [ `grep -c "failed to load 'identity_zero_1_int32' version 2: Internal: GPU instances not supported" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Server log ${SERVER_LOG} did not print model load failure\n***"
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_load_same_model_different_platform
for protocol in grpc http; do
    if [[ $protocol == "grpc" ]]; then
       export TRITONSERVER_USE_GRPC=1
    fi

    # The OS file system is more granular when determining modification time,
    # the modification timestamp is updated when the file content is changed in
    # place, but not updated when the file is copied or moved. With Triton, any
    # operation that changes a file is a modification. Thus, preparing the
    # models backward will test when a replacement model is having an earlier or
    # equal modification timestamp than the current model, Triton must still
    # detect the model is modified and proceed with model reload.
    for prep_order in normal reverse; do
        rm -fr models simple_float32_float32_float32
        mkdir models
        # Prepare two models of different platforms, but with the same name
        if [[ $prep_order == "normal" ]]; then
            # Prepare the TRT model first, then the pytorch model
            cp -r $DATADIR/qa_model_repository/plan_float32_float32_float32 models/simple_float32_float32_float32
            sed -i "s/plan_float32_float32_float32/simple_float32_float32_float32/" models/simple_float32_float32_float32/config.pbtxt
            cp -r $DATADIR/qa_model_repository/libtorch_float32_float32_float32 simple_float32_float32_float32
            sed -i "s/libtorch_float32_float32_float32/simple_float32_float32_float32/" simple_float32_float32_float32/config.pbtxt
        else
            # Prepare the pytorch model first, then the TRT model
            cp -r $DATADIR/qa_model_repository/libtorch_float32_float32_float32 simple_float32_float32_float32
            sed -i "s/libtorch_float32_float32_float32/simple_float32_float32_float32/" simple_float32_float32_float32/config.pbtxt
            cp -r $DATADIR/qa_model_repository/plan_float32_float32_float32 models/simple_float32_float32_float32
            sed -i "s/plan_float32_float32_float32/simple_float32_float32_float32/" models/simple_float32_float32_float32/config.pbtxt
        fi

        SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit \
                    --load-model=simple_float32_float32_float32 \
                    --exit-timeout-secs=5"
        SERVER_LOG="./inference_server_$LOG_IDX.log"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        rm -f $CLIENT_LOG
        set +e
        python $LC_TEST LifeCycleTest.test_load_same_model_different_platform >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID

        LOG_IDX=$((LOG_IDX+1))
    done

    unset TRITONSERVER_USE_GRPC
done

# Send HTTP request to control endpoint
rm -fr models config.pbtxt.*
mkdir models
for i in graphdef savedmodel onnx plan ; do
    cp -r $DATADIR/qa_model_repository/${i}_float32_float32_float32 models/.
done

# Polling enabled (default), control API should not work
# This test also keeps using "--model-store" to ensure backward compatibility
SERVER_ARGS="--model-store=`pwd`/models --repository-poll-secs=0 \
             --exit-timeout-secs=5 --strict-model-config=false \
             --model-control-mode=poll"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# unload API should return bad request
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/graphdef_float32_float32_float32/unload`
set -e
if [ "$code" == "200" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# the model should be available/ready
set +e
code=`curl -s -w %{http_code} localhost:8000/v2/models/graphdef_float32_float32_float32/ready`
set -e
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# remove model file so that if reload is triggered, model will become unavailable
rm models/graphdef_float32_float32_float32/*/*

# load API should return bad request
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/graphdef_float32_float32_float32/load`
set -e
if [ "$code" == "200" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# the model should be available/ready
set +e
code=`curl -s -w %{http_code} localhost:8000/v2/models/graphdef_float32_float32_float32/ready`
set -e
if [ "$code" != "200" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# Send HTTP request to invalid endpoints. This should be replaced by
# some more comprehensive fuzz attacks.
rm -fr models
mkdir models
for i in graphdef ; do
    cp -r $DATADIR/qa_model_repository/${i}_int32_int32_int32 models/.
done

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=none \
             --exit-timeout-secs=5"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/notapi/v2`
set -e
if [ "$code" != "404" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/notapi`
set -e
if [ "$code" != "404" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/notapi/foo`
set -e
if [ "$code" != "404" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_config_override
rm -fr models config.pbtxt.*
mkdir models
cp -r $DATADIR/qa_model_repository/onnx_float32_float32_float32 models/.
# Make only version 2 is valid version directory while config requests 1, 3
rm models/onnx_float32_float32_float32/1/*
rm models/onnx_float32_float32_float32/3/*

SERVER_ARGS="--model-repository=`pwd`/models --model-repository=`pwd`/models \
             --model-control-mode=explicit \
             --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_config_override >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

rm -f $CLIENT_LOG

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_file_override
rm -fr models config.pbtxt.*
mkdir models
cp -r $DATADIR/qa_model_repository/onnx_float32_float32_float32 models/.
# Make only version 2, 3 is valid version directory while config requests 1, 3
rm -rf models/onnx_float32_float32_float32/1

# Start with EXPLICIT mode and load onnx_float32_float32_float32
SERVER_ARGS="--model-repository=`pwd`/models \
             --model-control-mode=explicit \
             --load-model=onnx_float32_float32_float32 \
             --strict-model-config=false"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_file_override >>$CLIENT_LOG 2>&1
check_unit_test
python $LC_TEST LifeCycleTest.test_file_override_security >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

rm -f $CLIENT_LOG

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_shutdown_dynamic
rm -fr models config.pbtxt.*
mkdir models
cp -r ../custom_models/custom_zero_1_float32 models/. && \
    mkdir -p models/custom_zero_1_float32/1 && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching {}" >> config.pbtxt
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"5000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Server will be shutdown in test script, need to make PID available in script
SERVER_PID=$SERVER_PID python $LC_TEST LifeCycleTest.test_shutdown_dynamic >>$CLIENT_LOG 2>&1
check_unit_test
set -e

# check server log
if [ `grep -c "Model 'custom_zero_1_float32' (version 1) has 1 in-flight inferences" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect logging for model and in-flight inference count\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

rm -f $CLIENT_LOG

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_shutdown_sequence
rm -fr models config.pbtxt.*
mkdir models
cp -r ../custom_models/custom_sequence_int32 models/. && \
    mkdir -p models/custom_sequence_int32/1

SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Server will be shutdown in test script, need to make PID available in script
SERVER_PID=$SERVER_PID python $LC_TEST LifeCycleTest.test_shutdown_sequence >>$CLIENT_LOG 2>&1
check_unit_test
set -e

# check server log
if [ `grep -c "Model 'custom_sequence_int32' (version 1) has 2 in-flight inferences" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect logging for model having 2 in-flight inferences\n***"
    RET=1
fi
if [ `grep -c "Model 'custom_sequence_int32' (version 1) has 1 in-flight inferences" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect logging for model having 1 in-flight inference\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

rm -f $CLIENT_LOG

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_shutdown_ensemble
rm -fr models config.pbtxt.*
mkdir models
cp -r ensemble_zero_1_float32 models/. && \
    mkdir -p models/ensemble_zero_1_float32/1
cp -r ../custom_models/custom_zero_1_float32 models/. && \
    mkdir -p models/custom_zero_1_float32/1 && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching {}" >> config.pbtxt
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"5000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Server will be shutdown in test script, need to make PID available in script
SERVER_PID=$SERVER_PID python $LC_TEST LifeCycleTest.test_shutdown_ensemble >>$CLIENT_LOG 2>&1
check_unit_test
set -e

# check server log
if [ `grep -c "Model 'ensemble_zero_1_float32' (version 1) has 1 in-flight inferences" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect logging for model and in-flight inference count\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_load_gpu_limit
# dependency of the Python model to be used
pip install cuda-python
rm -fr models config.pbtxt.*
mkdir models
cp -r ../python_models/cuda_memory_consumer models/cuda_memory_consumer_1 && \
    cp -r ../python_models/cuda_memory_consumer models/cuda_memory_consumer_2

# Negative testing
SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --model-load-gpu-limit -1:0.6"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** unexpected start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
    kill $SERVER_PID
    wait $SERVER_PID
elif [ `grep -c "expects device ID >= 0, got -1" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect error on invalid device\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --model-load-gpu-limit 0:-0.4"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** unexpected start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
    kill $SERVER_PID
    wait $SERVER_PID
elif [ `grep -c "expects limit fraction to be in range \[0.0, 1.0\], got -0.4" $SERVER_LOG` == "0" ]; then
    echo -e "\n***\n*** Expect error on invalid fraction\n***"
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))

# Run server to stop model loading if > 60% of GPU 0 memory is used
SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --model-load-gpu-limit 0:0.6"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_load_gpu_limit >>$CLIENT_LOG 2>&1
check_unit_test
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_model_load_speedup
rm -rf models
mkdir models
MODEL_NAME="identity_zero_1_int32"
cp -r ${MODEL_NAME} models && mkdir -p models/${MODEL_NAME}/1
cp -r models/${MODEL_NAME} models/${MODEL_NAME}_1 && \
    sed -i "s/${MODEL_NAME}/${MODEL_NAME}_1/" models/${MODEL_NAME}_1/config.pbtxt
mv models/${MODEL_NAME} models/${MODEL_NAME}_2 && \
    sed -i "s/${MODEL_NAME}/${MODEL_NAME}_2/" models/${MODEL_NAME}_2/config.pbtxt
MODEL_NAME="identity_fp32"
cp -r ../python_models/${MODEL_NAME} models && (cd models/${MODEL_NAME} && \
    mkdir 1 && mv model.py 1 && \
    echo "    def initialize(self, args):" >> 1/model.py && \
    echo "        import time" >> 1/model.py && \
    echo "        time.sleep(10)" >> 1/model.py)
cp -r models/${MODEL_NAME} models/python_${MODEL_NAME}_1 && \
    sed -i "s/${MODEL_NAME}/python_${MODEL_NAME}_1/" models/python_${MODEL_NAME}_1/config.pbtxt
mv models/${MODEL_NAME} models/python_${MODEL_NAME}_2 && \
    sed -i "s/${MODEL_NAME}/python_${MODEL_NAME}_2/" models/python_${MODEL_NAME}_2/config.pbtxt

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_concurrent_model_load_speedup >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_model_load
rm -rf models models_v1 models_v2
mkdir models models_v2
cp -r identity_zero_1_int32 models/identity_model && \
    (cd models/identity_model && \
        mkdir 1 && \
        sed -i "s/identity_zero_1_int32/identity_model/" config.pbtxt)
cp -r ../python_models/identity_fp32 models_v2/identity_model && \
    (cd models_v2/identity_model && \
        mkdir 1 && mv model.py 1 && \
        sed -i "s/identity_fp32/identity_model/" config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_concurrent_model_load >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_model_load_unload
rm -rf models
mkdir models
cp -r identity_zero_1_int32 models && mkdir -p models/identity_zero_1_int32/1
cp -r ensemble_zero_1_float32 models && mkdir -p models/ensemble_zero_1_float32/1
cp -r ../custom_models/custom_zero_1_float32 models/. && \
    mkdir -p models/custom_zero_1_float32/1 && \
    (cd models/custom_zero_1_float32 && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"creation_delay_sec\"; value: { string_value: \"10\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_concurrent_model_load_unload >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_same_model_load_unload_stress
rm -rf models
mkdir models
cp -r identity_zero_1_int32 models && \
    (cd models/identity_zero_1_int32 && \
        mkdir 1 && \
        sed -i "s/string_value: \"10\"/string_value: \"0\"/" config.pbtxt)

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --model-load-thread-count=32 --log-verbose=2"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_concurrent_same_model_load_unload_stress >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    cat ./test_concurrent_same_model_load_unload_stress.statistics.log
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_model_instance_load_speedup
rm -rf models
mkdir models
MODEL_NAME="identity_fp32"
cp -r ../python_models/${MODEL_NAME} models/ && (cd models/${MODEL_NAME} && \
    mkdir 1 && mv model.py 1 && \
    echo "    def initialize(self, args):" >> 1/model.py && \
    echo "        import time" >> 1/model.py && \
    echo "        time.sleep(10)" >> 1/model.py)
rm models/${MODEL_NAME}/config.pbtxt

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $LC_TEST LifeCycleTest.test_concurrent_model_instance_load_speedup >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_concurrent_model_instance_load_sanity
rm -rf models
mkdir models
# Sanity check loading multiple instances in parallel for each supported backend
PARALLEL_BACKENDS="python onnx"
for backend in ${PARALLEL_BACKENDS} ; do
    model="${backend}_float32_float32_float32"
    model_dir="models/${model}"
    if [[ $backend == "python" ]]; then
      cp -r ../python_models/identity_fp32 ${model_dir}
      mkdir ${model_dir}/1 && mv ${model_dir}/model.py ${model_dir}/1
      rm ${model_dir}/config.pbtxt
    else
      mkdir models/${model}
      cp -r $DATADIR/qa_model_repository/${model}/1 models/${model}/1
    fi
done

SERVER_ARGS="--model-repository=`pwd`/models --model-control-mode=explicit --log-verbose=2"
SERVER_LOG="./inference_server_$LOG_IDX.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
PARALLEL_BACKENDS=${PARALLEL_BACKENDS} python $LC_TEST LifeCycleTest.test_concurrent_model_instance_load_sanity >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
