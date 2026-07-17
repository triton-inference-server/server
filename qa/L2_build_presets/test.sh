#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Validate the documented build-presets scenarios via build.py --dryrun (no GPU,
# container, or real build). Runs from the server source tree; see README.md.

TEST_LOG="build_presets_test.log"
TEST_REPORT="build_presets_test.report.xml"
RET=0
python3 -m pip install --quiet -r requirements.txt >/dev/null 2>&1 || true

# tee: stream to the console AND persist a full log for CI archiving.
# --junitxml=<path> writes machine-readable results (NOTE: the '=' is required;
# 'pytest --junitxml FILE.py' would treat FILE.py as the output path).
# PIPESTATUS[0] captures pytest's exit code, not tee's.
python3 -m pytest -v --log-cli-level=INFO --junitxml="$TEST_REPORT" build_presets_test.py 2>&1 | tee "$TEST_LOG"
RET=${PIPESTATUS[0]}

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
