#!/bin/bash
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NOBATCH_CLIENT_LOG_BASE="./nobatch_client"
NOBATCH_STRESS_TEST=nobatch_stress.py
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan python"}
source ../common/util.sh

for TARGET in $BACKENDS; do
    NOBATCH_CLIENT_LOG="$NOBATCH_CLIENT_LOG_BASE.$TARGET.log"
    python $NOBATCH_STRESS_TEST NoBatchStressTest.test_$TARGET > $NOBATCH_CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $NOBATCH_CLIENT_LOG
        echo -e "\n***\n*** Nobatch Test Failed\n***"
        RET=1
        echo "$RET" > RET.txt
    fi
    check_test_results $NOBATCH_CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $NOBATCH_CLIENT_LOG
        echo -e "\n***\n*** Nobatch Test Result Verification Failed\n***"
        RET=1
        echo "$RET" > RET.txt
    fi
done
