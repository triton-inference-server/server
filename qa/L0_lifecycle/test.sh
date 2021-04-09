#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
LC_TEST=lifecycle_test.py

DATADIR=/data/inferenceserver/${REPO_VERSION}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0
rm -fr *.log

LOG_IDX=0

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
sleep 10

set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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
sleep 10

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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
sleep 10

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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
sleep 10

rm -f $CLIENT_LOG
set +e
python $LC_TEST LifeCycleTest.test_parse_error_noexit >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

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
rm models/graphdef_float32_float32_float32/*/*

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

set +e
python $LC_TEST LifeCycleTest.test_parse_error_modelfail >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_parse_error_modelfail_nostrict >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_parse_error_no_model_config >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

LOG_IDX=$((LOG_IDX+1))

# LifeCycleTest.test_init_error_modelfail
rm -fr models models_0
mkdir models models_0
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

set +e
python $LC_TEST LifeCycleTest.test_init_error_modelfail >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_parse_error_model_no_version >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_parse_ignore_zero_prefixed_version >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
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

set +e
python $LC_TEST LifeCycleTest.test_parse_ignore_non_intergral_version >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

# LifeCycleTest.test_dynamic_model_load_unload_disabled
rm -fr models savedmodel_float32_float32_float32
mkdir models
for i in graphdef onnx plan ; do
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_load_unload_disabled >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_version_load_unload_disabled >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_model_modify >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

# LifeCycleTest.test_dynamic_file_delete
rm -fr models config.pbtxt.*
mkdir models
for i in savedmodel plan ; do
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

set +e
python $LC_TEST LifeCycleTest.test_dynamic_file_delete >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_polling >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_control >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_model_control >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_model_control_ensemble >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_multiple_model_repository_control_startup_models >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_model_repository_index >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $CLIENT_LOG 1
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

    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload_2 >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $CLIENT_LOG 1
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

    set +e
    python $LC_TEST LifeCycleTest.test_model_availability_on_reload_3 >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $CLIENT_LOG 1
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

set +e
python $LC_TEST LifeCycleTest.test_model_reload_fail >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
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

# LifeCycleTest.test_load_same_model_different_platform
for protocol in grpc http; do
    if [[ $protocol == "grpc" ]]; then
       export TRITONSERVER_USE_GRPC=1
    fi
    rm -fr models simple_float32_float32_float32
    mkdir models
    # Prepare two models of different platforms, but with the same name
    cp -r $DATADIR/qa_model_repository/plan_float32_float32_float32 models/simple_float32_float32_float32
    sed -i "s/plan_float32_float32_float32/simple_float32_float32_float32/" models/simple_float32_float32_float32/config.pbtxt
    cp -r $DATADIR/qa_model_repository/libtorch_float32_float32_float32 simple_float32_float32_float32
    sed -i "s/libtorch_float32_float32_float32/simple_float32_float32_float32/" simple_float32_float32_float32/config.pbtxt
    
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

    set +e
    python $LC_TEST LifeCycleTest.test_load_same_model_different_platform >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $CLIENT_LOG 1
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
if [ "$code" != "400" ]; then
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
if [ "$code" != "400" ]; then
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
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/notapi`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/models/notapi/foo`
set -e
if [ "$code" != "400" ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
