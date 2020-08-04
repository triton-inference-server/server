#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

RET=0

# Check image_client and perf_client
if [[ ! -x "triton_client/bin/image_client" ]]; then
    echo -e "*** image_client executable not present\n"
    RET=1
fi
if [[ ! -x "triton_client/bin/perf_client" ]]; then
    echo -e "*** perf_client executable not present\n"
    RET=1
fi

# Check static libraries
for l in libgrpcclient.so libgrpcclient_static.a libhttpclient.so libhttpclient_static.a; do
    if [[ ! -f "triton_client/lib/$l" ]]; then
        echo -e "*** library $l not present\n"
        RET=1
    fi
done

# Test a simple app using Triton gRPC API
client_lib=$(pwd)/triton_client/lib

# Test linking against the shared library
rm simple_grpc_client
g++ grpc_client.cc -o simple_grpc_client -Itriton_client/include \
  -L$(pwd)/triton_client/lib -I/workspace/builddir/grpc/include \
  -I/workspace/builddir/protobuf/include -lgrpcclient

if [ $? -eq 0 ]; then
    if [[ ! -x "./simple_grpc_client" ]]; then
        echo -e "*** simple_grpc_client executable not present\n"
        RET=1
    else
        ./simple_grpc_client
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** simple_grpc_client exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** simple_grpc_client exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

static_libs="$client_lib/libgrpcclient_static.a $client_lib/libgrpc++.a $client_lib/libgrpc.a \
             $client_lib/libgpr.a $client_lib/libcares.a $client_lib/libaddress_sorting.a $client_lib/libprotobuf.a \
             $client_lib/libcurl.a"

g++ grpc_client.cc $static_libs -o simple_grpc_client_static -Itriton_client/include  -I/workspace/builddir/grpc/include \
  -I/workspace/builddir/protobuf/include -lz -lssl -lcrypto -lpthread

if [ $? -eq 0 ]; then
    if [[ ! -x "./simple_grpc_client_static" ]]; then
        echo -e "*** simple_grpc_client_static executable not present\n"
        RET=1
    else
        ./simple_grpc_client_static
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** simple_grpc_client_static exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** simple_grpc_client_static exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

# Test a simple app using Triton HTTP API
g++ http_client.cc -o simple_http_client -Itriton_client/include \
  -L$(pwd)/triton_client/lib -lhttpclient

if [ $? -eq 0 ]; then
    if [[ ! -x "./simple_http_client" ]]; then
        echo -e "*** simple_http_client executable not present\n"
        RET=1
    else
        ./simple_http_client
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** simple_http_client exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** simple_http_client exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

g++ http_client.cc $client_lib/libhttpclient_static.a $client_lib/libcurl.a -o simple_http_client_static \
  -Itriton_client/include -lz -lssl -lcrypto -lpthread

if [ $? -eq 0 ]; then
    if [[ ! -x "./simple_http_client_static" ]]; then
        echo -e "*** simple_http_client executable not present\n"
        RET=1
    else
        ./simple_http_client_static
        if [ $? -eq 0 ]; then
            echo -e "\n***\n*** simple_http_client_static exited with 0 PASSED\n***"
        else
            echo -e "\n***\n*** simple_http_client_static exited with non-zero FAILED\n***"
            RET=1
        fi
    fi
else
    echo -e "\n***\n*** Client headers build FAILED\n***"
    RET=1
fi

# Check wheels
WHLVERSION=`cat /workspace/VERSION | sed 's/dev/\.dev0/'`
WHLS="tritonclientutils-${WHLVERSION}-py3-none-any.whl \
      tritongrpcclient-${WHLVERSION}-py3-none-any.whl \
      tritonhttpclient-${WHLVERSION}-py3-none-any.whl \
      tritonshmutils-${WHLVERSION}-py3-none-manylinux1_x86_64.whl"
for l in $WHLS; do
    if [[ ! -f "triton_client/python/$l" ]]; then
        echo -e "*** wheel $l not present\n"
        RET=1
    fi
done

# This test is running in a non-NVIDIA docker so we can configure the
# build without GPUs and make sure it works correctly.
cd /workspace/builddir && rm -fr client ../install/*
cmake -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF ../build
make -j16 client

CUDAFILES=`find /workspace/builddir/client/install -name *cuda* | wc -l`
if [ "$CUDAFILES" != "0" ]; then
    echo -e "*** unexpected CUDA files in TRITON_ENABLE_GPU=OFF build\n"
    RET=1
fi

SHMFILES=`find /workspace/builddir/client/install -name *shm* | wc -l`
if [ "$SHMFILES" != "7" ]; then
    echo -e "*** expected 7 SHM files in TRITON_ENABLE_GPU=OFF build, got $SHMFILES\n"
    RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
