#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG=client.log

# Install the tar file
rm -fr trtis_client
mkdir trtis_client
(cd trtis_client && tar xzvf /workspace/*.tar.gz)

# Build
cd trtis_client/src/cmake
cmake -DCMAKE_BUILD_TYPE=Release .
make -j16 trtis-clients

# There is no server running but can still check to make sure that the
# example application starts correctly.
set +e

# Shared HTTP
trtis-clients/install/bin/simple_client_shared > $CLIENT_LOG 2>&1
if [ $? -ne 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Unexpected Pass for simple_client_shared HTTP\n***"
    exit 1
fi

grep -c "Couldn't connect to server" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

# Shared GRPC
trtis-clients/install/bin/simple_client_shared -i grpc > $CLIENT_LOG 2>&1
if [ $? -ne 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Unexpected Pass for simple_client_shared GRPC\n***"
    exit 1
fi

grep -c "Connect Failed" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

# Static HTTP
trtis-clients/install/bin/simple_client_static > $CLIENT_LOG 2>&1
if [ $? -ne 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Unexpected Pass for simple_client_static HTTP\n***"
    exit 1
fi

grep -c "Couldn't connect to server" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

# Static GRPC
trtis-clients/install/bin/simple_client_static -i grpc > $CLIENT_LOG 2>&1
if [ $? -ne 1 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Unexpected Pass for simple_client_static GRPC\n***"
    exit 1
fi

grep -c "Connect Failed" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

set -e

echo -e "\n***\n*** Test Passed\n***"
