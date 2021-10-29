#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install the tar file
rm -fr triton_client
mkdir triton_client
(cd triton_client && tar xzvf /workspace/*.tar.gz)

set +e

RET=0

# Check image_client and perf_client
if [[ ! -x "triton_client/bin/image_client" ]]; then
    echo -e "*** image_client executable not present\n"
    RET=1
fi
if [[ ! -x "triton_client/bin/perf_analyzer" ]]; then
    echo -e "*** perf_analyzer executable is not present\n"
    RET=1
fi
if [[ ! -x "triton_client/bin/perf_client" ]]; then
    echo -e "*** perf_client link is not present\n"
    RET=1
fi

# Check static libraries
for l in libgrpcclient.so libgrpcclient_static.a libhttpclient.so libhttpclient_static.a; do
    if [[ ! -f "triton_client/lib/$l" ]]; then
        echo -e "*** library $l not present\n"
        RET=1
    fi
done

client_lib=$(pwd)/triton_client/lib
client_inc=$(pwd)/triton_client/include

# Test linking against the shared library
g++ grpc_test.cc -o grpc_test -I$client_inc -L$client_lib -lgrpcclient

if [ $? -eq 0 ]; then
    if [[ ! -x "./grpc_test" ]]; then
        echo -e "*** grpc_test executable not present\n"
        RET=1
    else
        ./grpc_test
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** grpc_test exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** grpc_test exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

#
# Test linking against static library
#

grpc_static_libs="-Wl,--start-group $client_lib/*.a -Wl,--end-group"

g++ grpc_test.cc $grpc_static_libs -o grpc_test_static -I$client_inc -lz -lssl -lcrypto -lpthread

if [ $? -eq 0 ]; then
    if [[ ! -x "./grpc_test_static" ]]; then
        echo -e "*** grpc_test_static executable not present\n"
        RET=1
    else
        ./grpc_test_static
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** grpc_test_static exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** grpc_test_static exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

#
# Test a simple app using Triton HTTP API
#

# Test linking against the shared library
g++ http_test.cc -o http_test -I$client_inc -L$client_lib -lhttpclient

if [ $? -eq 0 ]; then
    if [[ ! -x "./http_test" ]]; then
        echo -e "*** http_test executable not present\n"
        RET=1
    else
        ./http_test
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** http_test exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** http_test exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

g++ http_test.cc $client_lib/libhttpclient_static.a $client_lib/libcurl.a -o http_test_static \
  -I$client_inc -lz -lssl -lcrypto -lpthread

if [ $? -eq 0 ]; then
    if [[ ! -x "./http_test_static" ]]; then
        echo -e "*** http_test_static executable not present\n"
        RET=1
    else
        ./http_test_static
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** http_test_static exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** http_test_static exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

# Check wheels
WHLVERSION=`cat /workspace/TRITON_VERSION | sed 's/dev/\.dev0/'`
if [[ "aarch64" != $(uname -m) ]] ; then
    WHLS="tritonclient-${WHLVERSION}-py3-none-any.whl \
          tritonclient-${WHLVERSION}-py3-none-manylinux1_x86_64.whl"
else
    WHLS="tritonclient-${WHLVERSION}-py3-none-any.whl \
          tritonclient-${WHLVERSION}-py3-none-manylinux2014_aarch64.whl"
fi
for l in $WHLS; do
    if [[ ! -f "triton_client/python/$l" ]]; then
        echo -e "*** wheel $l not present\n"
        echo -e "*** available wheels in triton_client/python\n"
        ls -ltr triton_client/python
        RET=1
    fi
done

# Check wheel installation
python -c """import tritonclient; import tritonclient.grpc; import tritonclient.http; \
          import tritonclient.utils; import tritonclient.grpc.model_config_pb2; \
          import tritonclient.grpc.service_pb2; import tritonclient.grpc.service_pb2_grpc; \
          import tritonclient.utils.cuda_shared_memory; import tritonclient.utils.shared_memory"""
RET=$(($RET+$?))

EXECUTABLES="perf_analyzer perf_client"
for l in $EXECUTABLES; do
  if [ $(which -a $l | grep "/usr/local/bin/$l" | wc -l) -ne 1 ]; then
    which -a $l
    echo -e "*** $l executable not installed by tritonclient wheel\n"
    RET=1
  fi
done

# Check java client
if [[ ! -e "triton_client/java/java-api-0.0.1.jar" ]]; then
    echo -e "*** java-api-0.0.1.jar not present\n"
    RET=1
fi
if [[ ! -e "triton_client/java/examples/MemoryGrowthTest.jar" ]]; then
    echo -e "*** MemoryGrowthTest.jar not present\n"
    RET=1
fi
if [[ ! -e "triton_client/java/examples/SimpleInferClient.jar" ]]; then
    echo -e "*** SimpleInferClient.jar not present\n"
    RET=1
fi
if [[ ! -e "triton_client/java/examples/SimpleInferPerf.jar" ]]; then
    echo -e "*** SimpleInferPerf.jar not present\n"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
