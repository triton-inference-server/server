#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

pip3 install pytest-asyncio==0.23.8

RET=0

set +e

BINDING_TEST_LOG="./python_binding.log"
rm -f $BINDING_TEST_LOG
python -m pytest --junitxml=test_binding_report.xml test_binding.py > $BINDING_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $BINDING_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

API_TEST_LOG="./python_api.log"
rm -f $API_TEST_LOG
python -m pytest --junitxml=test_api_report.xml test_api.py > $API_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $API_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

FRONTEND_TEST_LOG="./python_kserve.log"
rm -f $FRONTEND_TEST_LOG
python -m pytest --junitxml=test_kserve.xml test_kserve.py > $FRONTEND_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $FRONTEND_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

LOG_CALLBACK_TEST_LOG="./python_logging_callback.log"
rm -f $LOG_CALLBACK_TEST_LOG
python -m pytest --junitxml=test_logging_callback_report.xml test_logging_callback.py > $LOG_CALLBACK_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $LOG_CALLBACK_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
