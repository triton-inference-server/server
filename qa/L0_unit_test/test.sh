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

export UNIT_TESTS="//src/..."
TEST_LOG="./unit_test.log"

# Need to have all libraries on a standard path since that is what is
# expected by bazel test.
cp /opt/tensorrtserver/lib/* /usr/lib/.

# Copy TensorRT plans into the unit test model repositories.
for modelpath in \
    testdata/autofill_sanity/too_many_inputs/1      \
        testdata/autofill_sanity/no_name_platform/1 \
        testdata/autofill_sanity/bad_input_type/1   \
        testdata/autofill_sanity/bad_input_dims/1   \
        testdata/autofill_sanity/unknown_input/1    \
        testdata/autofill_sanity/empty_config/1     \
        testdata/autofill_sanity/unknown_output/1   \
        testdata/autofill_sanity/bad_output_dims/1  \
        testdata/autofill_sanity/no_config/1        \
        testdata/autofill_sanity/bad_output_type/1  \
        testdata/autofill_sanity/too_few_inputs/1 ; do
    mkdir -p /workspace/src/backends/tensorrt/$modelpath
    cp /data/inferenceserver/qa_model_repository/plan_float32_float32_float32/1/model.plan \
       /workspace/src/backends/tensorrt/$modelpath/.
done

rm -f $TEST_LOG
RET=0

set +e

# Return code 3 indicates a test failure so ignore that failure as we
# use 'show_testlogs' to parse out more specific error messages.
(cd /workspace && \
        bazel test -c opt --verbose_failures --cache_test_results=no \
              --build_tests_only -- $(bazel query "tests($UNIT_TESTS)")) > $TEST_LOG 2>&1
BLDRET=$?
if [ $BLDRET -ne 0 ]; then
    RET=1
    if [ $BLDRET -ne 3 ]; then
      cat $TEST_LOG
      echo -e "\n***\n*** Failed to build\n***"
      exit 1
    fi
fi

grep "test\.log$" $TEST_LOG | /workspace/qa/common/show_testlogs
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
