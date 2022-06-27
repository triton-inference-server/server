#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RET=0

SIMPLE_AIO_INFER_CLIENT_PY=../clients/simple_http_aio_infer_client.py
SIMPLE_INFER_CLIENT_PY=../clients/simple_http_infer_client.py
TEST_CLIENT=../clients/simple_http_infer_client

NGINX_CONF=`pwd`/nginx.conf
CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

rm -f *.key *.crt ${CLIENT_LOG}.* server.log

# Generate valid CA
openssl genrsa -passout pass:1234 -des3 -out ca.key 4096
openssl req -passin pass:1234 -new -x509 -days 365 -key ca.key -out ca.crt -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Test/CN=Root CA"

# Generate valid Server Key/Cert
openssl genrsa -passout pass:1234 -des3 -out server.key 4096
openssl req -passin pass:1234 -new -key server.key -out server.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Server/CN=localhost"
openssl x509 -req -passin pass:1234 -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt

# Remove passphrase from the Server Key
openssl rsa -passin pass:1234 -in server.key -out server.key

# Generate valid Client Key/Cert
openssl genrsa -passout pass:1234 -des3 -out client.key 4096
openssl req -passin pass:1234 -new -key client.key -out client.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Client/CN=localhost"
openssl x509 -passin pass:1234 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt

# Remove passphrase from Client Key
openssl rsa -passin pass:1234 -in client.key -out client.key

# Create mutated client key (Make first char of each like capital)
cp client.key client2.key && sed -i "s/\b\(.\)/\u\1/g" client2.key
cp client.crt client2.crt && sed -i "s/\b\(.\)/\u\1/g" client2.crt

mv server.crt /etc/nginx/cert.crt
mv server.key /etc/nginx/cert.key

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Setup the new configuration for the proxy. The HTTPS traffic will be
# redirected to the running instance of server at localhost:8000
cp ${NGINX_CONF} /etc/nginx/sites-available/default

# Start the proxy server
service nginx restart

set +e

# Test basic inference with https
python $SIMPLE_INFER_CLIENT_PY -v -u localhost --ssl --key-file client.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.ssl_infer 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_infer
    RET=1
fi
python $SIMPLE_AIO_INFER_CLIENT_PY -v -u localhost --ssl --key-file client.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.ssl_infer.aio 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_infer.aio
    RET=1
fi

$TEST_CLIENT -v -u https://localhost:443 --key-file client.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.c++.ssl_infer 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.ssl_infer
    RET=1
fi

# Test basic inference on https without peer verification
python $SIMPLE_INFER_CLIENT_PY -v -u localhost --ssl --insecure >> ${CLIENT_LOG}.ssl_infer_insecure 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_infer_insecure
    RET=1
fi
python $SIMPLE_AIO_INFER_CLIENT_PY -v -u localhost --ssl --insecure >> ${CLIENT_LOG}.ssl_infer_insecure.aio 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_infer_insecure.aio
    RET=1
fi

$TEST_CLIENT -v -u https://localhost:443 --verify-host 0 --verify-peer 0 >> ${CLIENT_LOG}.c++.ssl_infer_insecure 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.ssl_infer_insecure
    RET=1
fi

# Test failure cases for SSL
# Try without SSL 
$SIMPLE_INFER_CLIENT_PY -v -u localhost >> ${CLIENT_LOG}.no_ssl_fail_infer 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.no_ssl_fail_infer
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi
$SIMPLE_AIO_INFER_CLIENT_PY -v -u localhost >> ${CLIENT_LOG}.no_ssl_fail_infer.aio 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.no_ssl_fail_infer.aio
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi

$TEST_CLIENT -v -u https://localhost:443 >> ${CLIENT_LOG}.c++.no_ssl_fail_infer 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.no_ssl_fail_infer
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi


# Try with incorrect key
$SIMPLE_INFER_CLIENT_PY -v -u localhost --ssl --key-file client2.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.ssl_wrong_key 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_wrong_key
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi
$SIMPLE_AIO_INFER_CLIENT_PY -v -u localhost --ssl --key-file client2.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.ssl_wrong_key.aio 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.ssl_wrong_key.aio
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi

$TEST_CLIENT -v -u https://localhost:443 --key-file client2.key --cert-file client.crt --ca-certs ca.crt >> ${CLIENT_LOG}.c++.ssl_wrong_key 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.c++.ssl_wrong_key
    echo -e "\n***\n*** Expected test failure\n***"
else
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Stop the proxy server
service nginx stop

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
