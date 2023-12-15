#!/usr/bin/env python

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

import sys

sys.path.append("../common")

import os
import random
import threading
import time
import unittest
from builtins import str
from functools import partial

import numpy as np
import sequence_util as su
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

TEST_SYSTEM_SHARED_MEMORY = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get("TEST_CUDA_SHARED_MEMORY", 0)))

USE_GRPC = os.environ.get("USE_GRPC", 1) != "0"
USE_HTTP = os.environ.get("USE_HTTP", 1) != "0"
assert USE_GRPC or USE_HTTP, "USE_GRPC or USE_HTTP must be non-zero"
if USE_GRPC and USE_HTTP:
    _protocols = ("http", "grpc")
elif USE_GRPC:
    _protocols = ("grpc",)
else:
    _protocols = ("http",)

BACKENDS = os.environ.get("BACKENDS", "graphdef savedmodel onnx plan custom python")
ENSEMBLES = bool(int(os.environ.get("ENSEMBLES", 1)))

NO_BATCHING = int(os.environ["NO_BATCHING"]) == 1
MODEL_INSTANCES = int(os.environ["MODEL_INSTANCES"])
IMPLICIT_STATE = int(os.environ["IMPLICIT_STATE"]) == 1

# Use initial state for implicit state
INITIAL_STATE_FILE = int(os.environ["INITIAL_STATE_FILE"]) == 1

_trials = ()
if NO_BATCHING:
    for backend in BACKENDS.split(" "):
        if backend != "custom":
            _trials += (backend + "_nobatch",)
elif os.environ["BATCHER_TYPE"] == "VARIABLE":
    for backend in BACKENDS.split(" "):
        if (backend != "libtorch") and (backend != "custom"):
            _trials += (backend,)
else:
    _trials = BACKENDS.split(" ")

# Add ensemble to the _trials
ENSEMBLE_PREFIXES = ["simple_", "sequence_", "fan_"]

if ENSEMBLES:
    res = []
    for trial in _trials:
        res.append(trial)
        if "custom" in trial:
            continue
        for ensemble_prefix in ENSEMBLE_PREFIXES:
            res.append(ensemble_prefix + trial)
    _trials = tuple(res)

_ragged_batch_supported_trials = list()
if "custom" in _trials:
    _ragged_batch_supported_trials = ("custom",)

# Not all models can be tested for ragged handling because the models
# don't deal well with non-size-1 shapes
_ragged_batch_not_supported_trials = list()
if os.environ["BATCHER_TYPE"] == "VARIABLE":
    if "custom" in _trials:
        _ragged_batch_not_supported_trials.append("custom")
    if "plan" in _trials:
        _ragged_batch_not_supported_trials.append("plan")
    if "onnx" in _trials:
        _ragged_batch_not_supported_trials.append("onnx")

_max_sequence_idle_ms = 5000


# Checks whether the provided model name belongs to an ensemble
# model.
def is_ensemble(model_name):
    for prefix in ENSEMBLE_PREFIXES:
        if model_name.startswith(prefix):
            return True
    return False


class SequenceBatcherTest(su.SequenceBatcherTestUtil):
    def get_datatype(self, trial):
        # Get the datatype to use based on what models are available (see test.sh)
        if "plan" in trial:
            return (np.float32,)
        if "custom" in trial:
            return (np.int32,)
        if "savedmodel" in trial:
            return (np.float32, np.bool_)
        if "graphdef" in trial:
            return (np.dtype(object), np.bool_)

        # Only test the string data type for ONNX and libtorch models in implicit state
        if IMPLICIT_STATE:
            if "onnx" in trial:
                return (np.dtype(object), np.int32, np.bool_)
            if NO_BATCHING:
                if "libtorch" in trial:
                    return (np.dtype(object), np.int32, np.bool_)

        return (np.int32, np.bool_)

    def get_expected_result(self, expected_result, value, trial, flag_str=None):
        # Adjust the expected_result for models that
        # could not implement the full accumulator. See
        # qa/common/gen_qa_sequence_models.py for more
        # information.
        if (
            (not NO_BATCHING and ("custom" not in trial))
            or ("graphdef" in trial)
            or ("plan" in trial)
            or ("onnx" in trial)
        ) or ("libtorch" in trial):
            expected_result = value
            if (flag_str is not None) and ("start" in flag_str):
                expected_result += 1
        return expected_result

    def get_expected_result_implicit(
        self, expected_result, value, trial, flag_str=None, dtype=None
    ):
        if dtype == np.dtype(object) and trial.startswith("onnx"):
            return value

        if INITIAL_STATE_FILE:
            # When the INITIAL_STATE_FILE is set the initial value
            # used for sequence will be 100 instead of zero and the
            # results will be offset by the same amount.
            return expected_result + 100
        else:
            return expected_result

    def test_simple_sequence(self):
        # Send one sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(45, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                45, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            5,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                (None, 3, None, None),
                                (None, 4, None, None),
                                (None, 5, None, None),
                                (None, 6, None, None),
                                (None, 7, None, None),
                                (None, 8, None, None),
                                ("end", 9, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: 9 * (idx + 1)}, 9 * (idx + 1), 9 * (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_length1_sequence(self):
        # Send a length-1 sequence and check for correct accumulator
        # result. The result should be returned immediately.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(42, 42, trial, "start,end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                42, 42, trial, "start,end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            99,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (("start,end", 42, None, None),),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: idx + 1}, (idx + 1), (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_batch_size(self):
        # Send sequence with a batch-size > 1 and check for error.

        # When 4 model instances the max-batch-size is 1 so can't test
        # since that gives a different error: "batch-size 2 exceeds
        # maximum batch size"
        if (MODEL_INSTANCES == 4) or NO_BATCHING:
            return

        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(10, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                10, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            27,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (("start", 1, None, None), ("end", 9, None, None)),
                            expected_result,
                            protocol,
                            batch_size=2,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request to model '{}' must specify "
                                            + "batch-size 1 due to requirements of sequence "
                                            + "batcher"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request to model '{}' must specify "
                                    + "batch-size 1 due to requirements of sequence "
                                    + "batcher"
                                ).format(model_name)
                            )
                        )

    def test_no_correlation_id(self):
        # Send sequence without correlation ID and check for error.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(10, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                10, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            0,  # correlation_id = 0
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (("start", 1, None, None), ("end", 9, None, None)),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request to model '{}' must specify a "
                                            + "non-zero or non-empty correlation ID"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request to model '{}' must specify a "
                                    + "non-zero or non-empty correlation ID"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_start(self):
        # Send sequence without start flag for never before seen
        # correlation ID. Expect failure.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )

                        expected_result = (
                            self.get_expected_result(6, 3, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                6, 3, trial, "end", dtype
                            )
                        )
                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            37469245,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (
                                (None, 1, None, None),
                                (None, 2, None, None),
                                ("end", 3, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        print(model_name + "-> " + ex.message())
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request for sequence 37469245 to "
                                            + "model '{}' must specify the START flag on the first "
                                            + "request of the sequence"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request for sequence 37469245 to "
                                    + "model '{}' must specify the START flag on the first "
                                    + "request of the sequence"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_start2(self):
        # Send sequence without start flag after sending a valid
        # sequence with the same correlation ID. Expect failure for
        # the second sequence.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(6, 3, trial, None)
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                6, 3, trial, None, dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            3,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                ("end", 3, None, None),
                                (None, 55, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_status(
                            model_name, {1: 3 * (idx + 1)}, 3 * (idx + 1), 3 * (idx + 1)
                        )
                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request for sequence 3 to model '{}' must "
                                            + "specify the START flag on the first request of "
                                            + "the sequence"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request for sequence 3 to model '{}' must "
                                    + "specify the START flag on the first request of "
                                    + "the sequence"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_end(self):
        # Send sequence without end flag. Use same correlation ID to
        # send another sequence. The first sequence will be ended
        # automatically but the second should complete successfully.
        for trial in _trials:
            # Run on different protocols.
            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)
                    # Skip bool type ensemble models
                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue
                    # For bool type control models, use int32 as I/O types
                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(51, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                51, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            4566,
                            (4000, None),
                            # (flag_str, value, (ls_ms, gt_ms), (pre_delay, post_delay))
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                ("start", 42, None, None),
                                ("end", 9, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: 4 * (idx + 1)}, 4 * (idx + 1), 4 * (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_half_batch(self):
        # Test model instances that together are configured with
        # total-batch-size 4. Send two equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 2 inferences.
        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3, 4), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (0, 9, 5, 13), dtype, 1
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 8)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    expected_result = (
                        self.get_expected_result(10, 4, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            10, 4, trial, "end", dtype
                        )
                    )

                    threads = []
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                987,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1, None),
                                    (None, 2, None),
                                    (None, 3, None),
                                    ("end", 4, None),
                                ),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(27, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            27, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                988,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 0, None),
                                    (None, 9, None),
                                    (None, 5, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 8}, 8, 8)
                    else:
                        stats_batch_size = 2 if MODEL_INSTANCES == 1 else 1
                        exec_cnt = 4 if MODEL_INSTANCES == 1 else 8
                        self.check_status(
                            model_name,
                            {stats_batch_size: 4 * min(2, MODEL_INSTANCES)},
                            exec_cnt,
                            8,
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_skip_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four sequences in parallel where
        # two sequences have shorter length so that padding must be
        # applied correctly for the longer sequences.
        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13, 14), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113, 1114), dtype, 3
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(50, 14, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            50, 14, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    (None, 13, None),
                                    ("end", 14, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4450, 1114, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4450, 1114, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    (None, 1113, None),
                                    ("end", 1114, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[1].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[0].start()
                    threads[2].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        # Batch size is 4 for the first two inferences and
                        # then 2 for the second two inferences. This is
                        # because we request the longer sequences first
                        # (threads 1 and 3) in slots 0 and 1 and so after
                        # shorter sequences are complete there are only slots
                        # 0 and 1 to execute.
                        if MODEL_INSTANCES == 1:
                            self.check_status(model_name, {2: 2, 4: 2}, 4, 12)
                        elif MODEL_INSTANCES == 2:
                            self.check_status(model_name, {2: 4, 1: 4}, 8, 12)
                        elif MODEL_INSTANCES == 4:
                            self.check_status(model_name, {1: 12}, 12, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_full_batch(self):
        # Test model instances together are configured with
        # total-batch-size 4. Send four equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences.
        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads = []
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(
                            model_name,
                            {(4 / MODEL_INSTANCES): (3 * MODEL_INSTANCES)},
                            3 * MODEL_INSTANCES,
                            12,
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_ragged_batch(self):
        # Test model instances that together are configured with
        # total-batch-size 4. The sequences use the different size
        # inputs and the inputs are *not* marked as allowing ragged
        # batch. Send four equal-length sequences in parallel and
        # make sure they don't get batched.

        # Only works with 1 model instance since want to test all
        # sequences batching together.
        if MODEL_INSTANCES != 1:
            return

        for trial in _ragged_batch_not_supported_trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0, tensor_shape=(2,)
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1, tensor_shape=(2,)
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2, tensor_shape=(1,)
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3, tensor_shape=(3,)
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6 * 2, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36 * 2, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (1,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336 * 3, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (3,),
                            },
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    time.sleep(3)
                    threads[3].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 9}, 9, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_ragged_batch_allowed(self):
        # Test model instances that together are configured with
        # total-batch-size 4. The sequences use the different size
        # inputs.  Send four equal-length sequences in parallel and
        # make sure they get batched appropriately even with size
        # differences.

        # Only works with 1 model instance since want to test all
        # sequences batching together.
        if MODEL_INSTANCES != 1:
            return

        for trial in _ragged_batch_supported_trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0, tensor_shape=(2,)
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1, tensor_shape=(2,)
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2, tensor_shape=(1,)
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3, tensor_shape=(3,)
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []

                    expected_result = (
                        self.get_expected_result(6 * 2, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6 * 2, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36 * 2, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36 * 2, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (1,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336 * 3, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336 * 3, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (3,),
                            },
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 3}, 3, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_backlog(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 5 equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences plus the 5th should go in the
        # backlog and then get handled once there is a free slot.
        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11112, 11113), dtype, 4
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(33336, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            33336, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11111, None),
                                    (None, 11112, None),
                                    ("end", 11113, None),
                                ),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 15}, 15, 15)
                    else:
                        if MODEL_INSTANCES == 1:
                            self.check_status(model_name, {4: 3, 1: 3}, 6, 15)
                        elif MODEL_INSTANCES == 2:
                            self.check_status(model_name, {2: 6, 1: 3}, 9, 15)
                        else:
                            self.check_status(model_name, {1: 15}, 15, 15)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_fill(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel, two of
        # which are shorter. Send 2 additional sequences that should
        # go into backlog but should immediately fill into the short
        # sequences.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111,), dtype, 4
                )
                precreated_shm5_handles = self.precreate_register_regions(
                    (22222,), dtype, 5
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 10
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 2
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(24, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            24, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 11, None), ("end", 13, None)),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(11111, 11111, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            11111, 11111, trial, "start,end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start,end", 11111, None),),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22222, 22222, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22222, 22222, trial, "start,end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1006,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start,end", 22222, None),),
                                expected_result,
                                precreated_shm5_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[4].start()
                    threads[5].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 3}, 3, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)
                        self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_fill_no_end(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel, two of
        # which are shorter. Send 2 additional sequences that should
        # go into backlog but should immediately fill into the short
        # sequences. One of those sequences is filled before it gets
        # its end request.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111,), dtype, 4
                )
                precreated_shm5_handles = self.precreate_register_regions(
                    (22222, 22223, 22224), dtype, 5
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 10
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 3
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(24, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            24, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 11, None), ("end", 13, None)),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(11111, 11111, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            11111, 11111, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start,end", 11111, None),),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(66669, 22224, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            66669, 22224, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1006,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 22222, None),
                                    (None, 22223, None),
                                    ("end", 22224, 2000),
                                ),
                                expected_result,
                                precreated_shm5_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(2)
                    threads[1].start()
                    time.sleep(2)
                    threads[2].start()
                    time.sleep(2)
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    time.sleep(2)
                    threads[5].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 14}, 14, 14)
                    else:
                        # Expecting 3 batch-size 4 inferences and then the
                        # 1006 sequence will follow 1003 (a different
                        # implementation could also follow 1002...)
                        self.check_status(model_name, {4: 3, 3: 2}, 5, 14)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)
                        self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_same_correlation_id(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 equal-length sequences in
        # parallel and make sure they get completely batched into
        # batch-size 4 inferences. Send a 5th with the same
        # correlation ID as one of the first four.
        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )

                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 2
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[4].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 14}, 14, 14)
                    else:
                        if MODEL_INSTANCES != 4:
                            batch_exec = {
                                (4 / MODEL_INSTANCES): (3 * MODEL_INSTANCES),
                                1: 2,
                            }
                        else:
                            batch_exec = {1: (3 * MODEL_INSTANCES) + 2}
                        self.check_status(
                            model_name, batch_exec, (3 * MODEL_INSTANCES) + 2, 14
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_same_correlation_id_no_end(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel and
        # make sure they get completely batched into batch-size 4
        # inferences. One of the sequences is shorter and does not
        # have an end marker but has same correlation ID as the 5th
        # sequence. We expect that short sequence to get ended early
        # (because of the same correlation ID) and make room for the
        # 5th sequence.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for both sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 16
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(4, 3, trial, None, dtype)
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None), (None, 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(48, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            48, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(448, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            448, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4448, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4448, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):
                        # Requests do not get batched for the ensemble model
                        self.check_status(model_name, {1: 16}, 16, 16)
                    else:
                        self.check_status(model_name, {4: 4}, 4, 16)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_sequence_timeout(self):
        # Test model instances together are configured with
        # total-max-batch-size 4. Send 4 sequences in parallel and
        # make sure they get completely batched into batch-size 4
        # inferences. One of the sequences has a long delay that
        # causes it to timeout and that allows a 5th sequence to come
        # out of the backlog and finish. The timed-out sequence will
        # then send the delayed inference but it will appear as a new
        # sequence and so fail because it doesn't have the START flag.

        # Only works with 1 model instance since otherwise an instance
        # can run ahead and handle more work than expected (leads to
        # intermittent failures)
        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # Skip bool type ensemble models
                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain all
                    # inferences for all sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(4, 3, trial, None, dtype)
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1, None),
                                    (None, 3, _max_sequence_idle_ms + 1000),
                                ),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(48, 13, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            48, 13, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, _max_sequence_idle_ms / 2),
                                    (None, 12, _max_sequence_idle_ms / 2),
                                    ("end", 13, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(448, 113, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            448, 113, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 111, None),
                                    (None, 112, _max_sequence_idle_ms / 2),
                                    (None, 112, _max_sequence_idle_ms / 2),
                                    ("end", 113, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4448, 1113, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4448, 1113, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 1111, None),
                                    (None, 1112, _max_sequence_idle_ms / 2),
                                    (None, 1112, _max_sequence_idle_ms / 2),
                                    ("end", 1113, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except Exception as ex:
                    for prefix in ENSEMBLE_PREFIXES:
                        if model_name.startswith(prefix):
                            base_model_name = model_name[(len(prefix)) :]
                            self.assertTrue(
                                ex.message().startswith(
                                    str(
                                        "in ensemble '{}', "
                                        + "inference request for sequence 1001 to "
                                        + "model '{}' must specify the START flag on the first "
                                        + "request of the sequence"
                                    ).format(model_name, base_model_name)
                                )
                            )
                            return
                    self.assertTrue(
                        ex.message().startswith(
                            str(
                                "inference request for sequence 1001 to "
                                + "model '{}' must specify the START flag on the first "
                                + "request of the sequence"
                            ).format(model_name)
                        )
                    )
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_queue_delay_no_min_util(self):
        # Test model that have set max queue delay but minimum slot utilization
        # is 0. Send 2 sequences in parallel and make sure they get completely
        # batched into batch-size 2 inferences. The first sequence only has one
        # request while the second sequence has two, so expecting the second
        # execution to be a batch of 'null, seq 2'. The executions should not be
        # waited.

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain 2 sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (2000, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (2000, None),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_queue_delay_half_min_util(self):
        # Test model that have set max queue delay but minimum slot utilization
        # is 0.5. Send 2 sequences in parallel and make sure they get completely
        # batched into batch-size 2 inferences. The first sequence only has one
        # request while the second sequence has two, so expecting the second
        # execution to be a batch of 'null, seq 2'. The second execution should
        # be waited until the max queue delay is exceeded for sequence 2.

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype) + "_half"
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain 2 sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (2000, None),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (4000, 3000),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_queue_delay_full_min_util(self):
        # Test model that have set max queue delay but minimum slot utilization
        # is 1. Send 2 sequences in parallel and make sure they get completely
        # batched into batch-size 2 inferences. The first sequence only has one
        # request while the second sequence has two, so expecting the second
        # execution to be a batch of 'null, seq 2'. Both executions should be
        # waited until the max queue delay is exceeded.

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype) + "_full"
                # For bool type control models, use int32 as I/O types
                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    # Need scheduler to wait for queue to contain 2 sequences.
                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (4000, 3000),
                                # (flag_str, value, pre_delay_ms)
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (6000, 5000),
                                # (flag_str, value, pre_delay_ms)
                                (
                                    ("start", 11, None),
                                    (None, 12, 2000),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)


class SequenceBatcherRequestTimeoutTest(su.SequenceBatcherTestUtil):
    def setUp(self):
        super(SequenceBatcherRequestTimeoutTest, self).setUp()
        # By default, find tritonserver on "localhost", but can be overridden
        # with TRITONSERVER_IPADDR envvar
        self.server_address_ = (
            os.environ.get("TRITONSERVER_IPADDR", "localhost") + ":8001"
        )

        # Prepare input and expected output based on the model and
        # the infer sequence sent for testing. If the test is to be extended
        # for different sequence and model, then proper grouping should be added
        self.model_name_ = "custom_sequence_int32_timeout"
        self.tensor_data_ = np.ones(shape=[1, 1], dtype=np.int32)
        self.inputs_ = [grpcclient.InferInput("INPUT0", [1, 1], "INT32")]
        self.inputs_[0].set_data_from_numpy(self.tensor_data_)
        self.expected_out_seq_ = [
            ("OUTPUT0", self.tensor_data_),
            ("OUTPUT0", self.tensor_data_),
            ("OUTPUT0", self.tensor_data_),
        ]

    def send_sequence_with_timeout(
        self, seq_id, callback, timeout_us=3000000, request_pause_sec=0
    ):
        with grpcclient.InferenceServerClient(self.server_address_) as triton_client:
            triton_client.start_stream(callback=callback)
            triton_client.async_stream_infer(
                self.model_name_,
                self.inputs_,
                sequence_id=seq_id,
                sequence_start=True,
                timeout=timeout_us,
            )
            if request_pause_sec != 0:
                time.sleep(request_pause_sec)
            triton_client.async_stream_infer(
                self.model_name_, self.inputs_, sequence_id=seq_id, timeout=timeout_us
            )
            if request_pause_sec != 0:
                time.sleep(request_pause_sec)
            triton_client.async_stream_infer(
                self.model_name_,
                self.inputs_,
                sequence_id=seq_id,
                sequence_end=True,
                timeout=timeout_us,
            )

    def test_request_timeout(self):
        # Test long running model that receives requests with shorter timeout,
        # expect the timeout will only be expired on backlog sequence and reject
        # all requests of the sequence once expired.
        # Sending two sequences while the model can only process one sequence
        # at a time. Each model execution takes 5 second and all requests have
        # 3 second timeout, so the second sequence will be rejected.

        # correlation ID is 1-index
        seq1_res = []
        seq2_res = []
        seq1_callback = lambda result, error: seq1_res.append((result, error))
        seq2_callback = lambda result, error: seq2_res.append((result, error))

        # send sequence with 1s interval to ensure processing order
        threads = []
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(1, seq1_callback)
            )
        )
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(2, seq2_callback)
            )
        )
        threads[0].start()
        time.sleep(1)
        threads[1].start()
        for t in threads:
            t.join()

        for idx in range(len(seq1_res)):
            result, error = seq1_res[idx]
            self.assertIsNone(
                error,
                "Expect successful inference for sequence 1 requests, got error: {}".format(
                    error
                ),
            )
            out = result.as_numpy(self.expected_out_seq_[idx][0])
            expected_out = self.expected_out_seq_[idx][1]
            np.testing.assert_allclose(
                out,
                expected_out,
                err_msg="Unexpected output tensor: expect {}, got {}".format(
                    expected_out, out
                ),
            )

        for _, error in seq2_res:
            self.assertIsNotNone(error, "Expect error for sequence 2 requests")
            with self.assertRaisesRegex(
                InferenceServerException,
                "timeout of the corresponding sequence has been expired",
                msg="Unexpected error: {}".format(error),
            ):
                raise error

    def test_send_request_after_timeout(self):
        # Similar to test_request_timeout, but the sequence to be timed out
        # will send the last request after the sequence has been timed out,
        # and expecting server to return error regarding sending request of
        # an untracked sequence

        seq1_res = []
        seq2_res = []
        seq1_callback = lambda result, error: seq1_res.append((result, error))
        seq2_callback = lambda result, error: seq2_res.append((result, error))

        threads = []
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(1, seq1_callback)
            )
        )
        # Each request will be sent with a pause, so the third request
        # will be sent after the sequence has been timed out
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout,
                args=(2, seq2_callback),
                kwargs={"request_pause_sec": 2},
            )
        )
        threads[0].start()
        time.sleep(1)
        threads[1].start()
        for t in threads:
            t.join()

        # Check error message of the last request and the rest
        # separately
        for _, error in seq2_res[0:-1]:
            self.assertIsNotNone(error, "Expect error for sequence 2 requests")
            with self.assertRaisesRegex(
                InferenceServerException,
                "timeout of the corresponding sequence has been expired",
                msg="Unexpected error: {}".format(error),
            ):
                raise error
        _, last_err = seq2_res[-1]
        self.assertIsNotNone(last_err, "Expect error for sequence 2 requests")
        with self.assertRaisesRegex(
            InferenceServerException,
            "must specify the START flag on the first request",
            msg="Unexpected error: {}".format(last_err),
        ):
            raise last_err


class SequenceBatcherPreserveOrderingTest(su.SequenceBatcherTestUtil):
    def setUp(self):
        super().setUp()
        # By default, find tritonserver on "localhost", but can be overridden
        # with TRITONSERVER_IPADDR envvar
        self.server_address_ = (
            os.environ.get("TRITONSERVER_IPADDR", "localhost") + ":8001"
        )

        # Prepare input and expected output based on the model and
        # the infer sequence sent for testing. If the test is to be extended
        # for different sequence and model, then proper grouping should be added
        self.model_name_ = "sequence_py"
        self.tensor_data_ = np.ones(shape=[1, 1], dtype=np.int32)
        self.inputs_ = [grpcclient.InferInput("INPUT0", [1, 1], "INT32")]
        self.inputs_[0].set_data_from_numpy(self.tensor_data_)
        self.triton_client = grpcclient.InferenceServerClient(self.server_address_)

        # Atomic request ID for multi-threaded inference
        self.request_id_lock = threading.Lock()
        self.request_id = 1

    def send_sequence(self, seq_id, seq_id_map, req_id_map):
        if seq_id not in seq_id_map:
            seq_id_map[seq_id] = []

        start, middle, end = (True, False), (False, False), (False, True)
        # Send sequence with 1 start, 1 middle, and 1 end request
        seq_flags = [start, middle, end]
        for start_flag, end_flag in seq_flags:
            # Introduce random sleep to better interweave requests from different sequences
            time.sleep(random.uniform(0.0, 1.0))

            # Serialize sending requests to ensure ordered request IDs
            with self.request_id_lock:
                req_id = self.request_id
                self.request_id += 1

                # Store metadata to validate results later
                req_id_map[req_id] = seq_id
                seq_id_map[seq_id].append(req_id)

                self.triton_client.async_stream_infer(
                    self.model_name_,
                    self.inputs_,
                    sequence_id=seq_id,
                    sequence_start=start_flag,
                    sequence_end=end_flag,
                    timeout=None,
                    request_id=str(req_id),
                )

    def _test_sequence_ordering(self, preserve_ordering, decoupled):
        # 1. Send a few grpc streaming sequence requests to the model.
        # 2. With grpc streaming, the model should receive the requests in
        #    the same order they are sent from client, and the client should
        #    receive the responses in the same order sent back by the
        #    model/server. With sequence scheduler, the requests for each sequence should be routed to the same model
        #    instance, and no two requests from the same sequence should
        #    get batched together.
        # 3. With preserve_ordering=False, we may get the responses back in a different
        #    order than the requests, but with grpc streaming we should still expect responses for each sequence to be ordered.
        # 4. Assert that the sequence values are ordered, and that the response IDs per sequence are ordered
        class SequenceResult:
            def __init__(self, seq_id, result, request_id):
                self.seq_id = seq_id
                self.result = result
                self.request_id = int(request_id)

        def full_callback(sequence_dict, sequence_list, result, error):
            # We expect no model errors for this test
            if error:
                self.assertTrue(False, error)

            # Gather all the necessary metadata for validation
            request_id = int(result.get_response().id)
            sequence_id = request_id_map[request_id]
            # Overall list of results in the order received, regardless of sequence ID
            sequence_list.append(SequenceResult(sequence_id, result, request_id))
            # Ordered results organized by their seq IDs
            sequence_dict[sequence_id].append(result)

        # Store ordered list in which responses are received by client
        sequence_list = []
        # Store mapping of sequence ID to response results
        sequence_dict = {}
        # Store mapping of sequence ID to request IDs and vice versa
        sequence_id_map = {}
        request_id_map = {}

        # Start stream
        seq_callback = partial(full_callback, sequence_dict, sequence_list)
        self.triton_client.start_stream(callback=seq_callback)

        # Send N sequences concurrently
        threads = []
        num_sequences = 10
        for i in range(num_sequences):
            # Sequence IDs are 1-indexed
            sequence_id = i + 1
            # Add a result list and callback for each sequence
            sequence_dict[sequence_id] = []
            threads.append(
                threading.Thread(
                    target=self.send_sequence,
                    args=(sequence_id, sequence_id_map, request_id_map),
                )
            )

        # Start all sequence threads
        for t in threads:
            t.start()

        # Wait for threads to return
        for t in threads:
            t.join()

        # Block until all requests are completed
        self.triton_client.stop_stream()

        # Make sure some inferences occurred and metadata was collected
        self.assertGreater(len(sequence_dict), 0)
        self.assertGreater(len(sequence_list), 0)

        # Validate model results are sorted per sequence ID (model specific logic)
        print(f"=== {preserve_ordering=} {decoupled=} ===")
        print("Outputs per Sequence:")
        for seq_id, sequence in sequence_dict.items():
            seq_outputs = [
                result.as_numpy("OUTPUT0").flatten().tolist() for result in sequence
            ]
            print(f"{seq_id}: {seq_outputs}")
            self.assertEqual(seq_outputs, sorted(seq_outputs))

        # Validate request/response IDs for each response in a sequence is sorted
        # This should be true regardless of preserve_ordering or not
        print("Request IDs per Sequence:")
        for seq_id in sequence_id_map:
            per_seq_request_ids = sequence_id_map[seq_id]
            print(f"{seq_id}: {per_seq_request_ids}")
            self.assertEqual(per_seq_request_ids, sorted(per_seq_request_ids))

        # Validate results are sorted in request order if preserve_ordering is True
        if preserve_ordering:
            request_ids = [s.request_id for s in sequence_list]
            print(f"Request IDs overall:\n{request_ids}")
            sequence_ids = [s.seq_id for s in sequence_list]
            print(f"Sequence IDs overall:\n{sequence_ids}")
            self.assertEqual(request_ids, sorted(request_ids))

        # Assert some dynamic batching of requests was done
        stats = self.triton_client.get_inference_statistics(
            model_name=self.model_name_, headers={}, as_json=True
        )
        model_stats = stats["model_stats"][0]
        self.assertEqual(model_stats["name"], self.model_name_)
        self.assertLess(
            int(model_stats["execution_count"]), int(model_stats["inference_count"])
        )

    def test_sequence_with_preserve_ordering(self):
        self.model_name_ = "seqpy_preserve_ordering_nondecoupled"
        self._test_sequence_ordering(preserve_ordering=True, decoupled=False)

    def test_sequence_without_preserve_ordering(self):
        self.model_name_ = "seqpy_no_preserve_ordering_nondecoupled"
        self._test_sequence_ordering(preserve_ordering=False, decoupled=False)

    # FIXME [DLIS-5280]: This may fail for decoupled models if writes to GRPC
    # stream are done out of order in server, so disable test for now.
    # def test_sequence_with_preserve_ordering_decoupled(self):
    #    self.model_name_ = "seqpy_preserve_ordering_decoupled"
    #    self._test_sequence_ordering(preserve_ordering=True, decoupled=True)

    # FIXME [DLIS-5280]
    # def test_sequence_without_preserve_ordering_decoupled(self):
    #    self.model_name_ = "seqpy_no_preserve_ordering_decoupled"
    #    self._test_sequence_ordering(preserve_ordering=False, decoupled=True)


if __name__ == "__main__":
    unittest.main()
