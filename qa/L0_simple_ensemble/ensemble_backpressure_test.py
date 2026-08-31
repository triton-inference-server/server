#!/usr/bin/env python3

# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import queue
import threading
import time
import unittest
from contextlib import ExitStack
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

SERVER_URL = "localhost:8001"
DEFAULT_RESPONSE_TIMEOUT = 60
EXPECTED_INFER_OUTPUT = 0.5
MODEL_ENSEMBLE_PARALLEL_FAILED_ENQUEUE = "ensemble_parallel_step_failed_enqueue"
EXPECTED_PARALLEL_FAILED_ENQUEUE_OUTPUT = 4.0

NUM_REQUESTS = 16
NUM_RESPONSES_PER_REQUEST = 8


# ---------------------------------------------------------------------------
# Debug instrumentation
#
# The concurrent backpressure test intermittently fails with "expected 8
# responses, got 0", but that assertion masks *why* Triton ended the stream
# early. The helpers below record, for every streamed message, whether it was a
# data response, an empty final-only response, or an error (with its gRPC
# status/message), plus timing and a full per-stream timeline -- so a failing
# run tells us the actual reason the response came back from the server.
#
# Set TRITON_TEST_DEBUG=0 to silence the extra logging.
# ---------------------------------------------------------------------------
_DEBUG_ENABLED = os.environ.get("TRITON_TEST_DEBUG", "1") != "0"
_DEBUG_T0 = time.time()
_DEBUG_LOCK = threading.Lock()


def _dbg(tag, msg):
    """Thread-safe, timestamped debug line to stderr (captured in the client log)."""
    if not _DEBUG_ENABLED:
        return
    now = time.time()
    line = f"[DBG {now:.6f} +{now - _DEBUG_T0:8.3f}s tid={threading.get_ident()} {tag}] {msg}"
    with _DEBUG_LOCK:
        print(line, file=sys.stderr, flush=True)


def _final_flag(response):
    """bool value of the triton_final_response parameter, or None if absent/unreadable."""
    try:
        final = response.parameters.get("triton_final_response")
        return bool(final.bool_param) if final is not None else None
    except Exception:
        return None


def _describe_error(error):
    """Human-readable reason string for an InferenceServerException from Triton."""
    try:
        return f"ERROR status={error.status()!r} msg={error.message()!r} full={str(error)!r}"
    except Exception as exc:
        return f"ERROR (undecodable: {exc!r}) raw={error!r}"


def _describe_result(result):
    """Human-readable summary of a data / empty-final response from Triton."""
    try:
        response = result.get_response()
        n_out = len(response.outputs)
        out_names = [o.name for o in response.outputs]
        value = None
        if n_out > 0:
            try:
                arr = result.as_numpy("OUT")
                value = None if arr is None else float(np.squeeze(arr))
            except Exception:
                value = "<unreadable>"
        return (
            f"RESPONSE id={response.id!r} outputs={n_out} names={out_names} "
            f"final={_final_flag(response)} OUT={value}"
        )
    except Exception as exc:
        return f"RESPONSE (undecodable: {exc!r})"


def _errs_brief(errors):
    """Compact (status, message) list for use in assertion messages."""
    brief = []
    for err in errors:
        try:
            brief.append((str(err.status()), err.message()))
        except Exception:
            brief.append(("<undecodable>", repr(err)))
    return brief


class UserData:
    def __init__(self, tag=None):
        self._response_queue = queue.Queue()
        # Debug context: a label for this stream and a full timeline of what
        # Triton sent back, so a failure can be explained after the fact.
        self.tag = tag
        self.events = []  # list of (elapsed_s, kind, description)
        self._start = time.time()


def callback(user_data, result, error):
    now = time.time()
    elapsed = now - getattr(user_data, "_start", _DEBUG_T0)
    tag = getattr(user_data, "tag", None)
    if error is not None:
        desc = _describe_error(error)
        _dbg(f"CB {tag}", f"t+{elapsed:7.3f}s {desc}")
        try:
            user_data.events.append((round(elapsed, 3), "error", desc))
        except Exception:
            pass
        user_data._response_queue.put(error)
    else:
        desc = _describe_result(result)
        _dbg(f"CB {tag}", f"t+{elapsed:7.3f}s {desc}")
        try:
            user_data.events.append((round(elapsed, 3), "response", desc))
        except Exception:
            pass
        user_data._response_queue.put(result)


def prepare_infer_args(input_value, enable_batching=False):
    """
    Create InferInput/InferRequestedOutput lists
    """
    if enable_batching:
        input_data = np.array([[input_value]], dtype=np.int32)
    else:
        input_data = np.array([input_value], dtype=np.int32)
    infer_input = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
    infer_input[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    return infer_input, outputs


def collect_responses(user_data, timeout=DEFAULT_RESPONSE_TIMEOUT):
    """
    Collect responses from user_data until the final response flag is seen.

    Returns (errors, responses); the signature is unchanged so existing callers
    keep working. Rich per-message diagnostics (empty-final vs data vs error,
    latency, and the full timeline) are emitted via _dbg so we can see WHY Triton
    ended a stream.
    """
    tag = getattr(user_data, "tag", None)
    errors = []
    responses = []
    saw_empty_final = False
    first_latency = None
    t_start = time.time()
    msg_idx = 0

    while True:
        try:
            result = user_data._response_queue.get(timeout=timeout)
        except queue.Empty:
            # No message arrived within `timeout`s. This is the fingerprint of a
            # stalled/starved stream (vs. an immediate error or empty final).
            _dbg(
                f"COLLECT {tag}",
                f"TIMEOUT after {timeout}s waiting for msg #{msg_idx + 1}; "
                f"so far responses={len(responses)} errors={len(errors)} "
                f"saw_empty_final={saw_empty_final}; timeline={user_data.events}",
            )
            raise Exception(
                f"[{tag}] No response received within {timeout} seconds "
                f"(got {len(responses)} data responses, {len(errors)} errors so far). "
                f"timeline={user_data.events}"
            )

        msg_idx += 1
        if first_latency is None:
            first_latency = time.time() - t_start

        if isinstance(result, InferenceServerException):
            errors.append(result)
            _dbg(
                f"COLLECT {tag}",
                f"msg#{msg_idx} is ERROR -> stream terminates: {_describe_error(result)}",
            )
            # error responses are final - stream terminates
            break

        response = result.get_response()
        n_out = len(response.outputs)
        final = _final_flag(response)
        # Add response to list if it has data (not empty final-only response)
        if n_out > 0:
            responses.append(result)
        elif final:
            saw_empty_final = True
        _dbg(
            f"COLLECT {tag}",
            f"msg#{msg_idx} outputs={n_out} final={final} kept={len(responses)} "
            f"saw_empty_final={saw_empty_final}",
        )

        # Check if this is the final response
        if final:
            break

    total = time.time() - t_start
    _dbg(
        f"COLLECT {tag}",
        f"DONE responses={len(responses)} errors={len(errors)} "
        f"saw_empty_final={saw_empty_final} "
        f"first_latency={None if first_latency is None else round(first_latency, 3)} "
        f"total={round(total, 3)}s",
    )
    return errors, responses


class EnsembleBackpressureTest(tu.TestResultCollector):
    """
    Tests for ensemble backpressure feature (max_inflight_requests).
    """

    def _run_inference(
        self,
        model_name,
        expected_responses_per_request,
        num_concurrent_requests=1,
        stream_timeout=None,
        channel_args=None,
    ):
        """
        Send num_concurrent_requests streaming requests to model_name, each expecting
        expected_responses_per_request responses. Verify all complete with correct data.

        stream_timeout (seconds) is forwarded to start_stream; None (the default)
        means no client-side stream timeout, matching the original test.
        channel_args (list of (name, value)) is forwarded to each client's gRPC
        channel; None (the default) matches the original test. Passing
        [("grpc.use_local_subchannel_pool", 1)] gives every client its own TCP
        connection instead of sharing pooled connections.
        """
        # Tag each stream so the debug timeline identifies which request is which.
        user_datas = [
            UserData(tag=f"{model_name}#req{i}")
            for i in range(num_concurrent_requests)
        ]

        with ExitStack() as stack:
            clients = [
                stack.enter_context(
                    grpcclient.InferenceServerClient(
                        SERVER_URL, channel_args=channel_args
                    )
                )
                for _ in range(num_concurrent_requests)
            ]

            inputs, outputs = prepare_infer_args(expected_responses_per_request, True)

            _dbg(
                "RUN",
                f"model={model_name} concurrent={num_concurrent_requests} "
                f"expected_per_request={expected_responses_per_request} "
                f"stream_timeout={stream_timeout} channel_args={channel_args} "
                f"-> starting streams at wall={time.time():.6f}",
            )

            # Start all concurrent requests
            for i in range(num_concurrent_requests):
                clients[i].start_stream(
                    callback=partial(callback, user_datas[i]),
                    stream_timeout=stream_timeout,
                )
                clients[i].async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

            # Collect and verify responses for all requests
            for i, ud in enumerate(user_datas):
                errors, responses = collect_responses(ud)

                # One combined diagnostic so a failure never hides the real reason
                # (error status, empty final, or missing data) or the timing.
                diag = (
                    f"[{ud.tag}] responses={len(responses)} "
                    f"errors={_errs_brief(errors)} timeline={ud.events}"
                )
                if len(responses) != expected_responses_per_request or errors:
                    _dbg("RUN", f"MISMATCH {diag}")

                # Check errors FIRST: if Triton returned an error, surface its
                # gRPC status/message instead of the misleading "got 0" count.
                self.assertEqual(
                    len(errors),
                    0,
                    f"Request {i}: Triton returned error(s) {_errs_brief(errors)}. {diag}",
                )
                self.assertEqual(
                    len(responses),
                    expected_responses_per_request,
                    f"Request {i}: expected {expected_responses_per_request} responses, "
                    f"got {len(responses)}. {diag}",
                )
                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    # output shape is [batch_size, 1]; extract scalar for comparison.
                    value = float(output[0][0])
                    self.assertAlmostEqual(
                        value,
                        EXPECTED_INFER_OUTPUT,
                        places=5,
                        msg=f"Request {i} response {idx}: expected "
                        f"{EXPECTED_INFER_OUTPUT}, got {value}",
                    )

            # Stop all streams
            for client in clients:
                client.stop_stream()

    # ------------------------------------------------------------------
    # Experiment probes (diagnostic; remove once root cause is confirmed).
    # These isolate WHY the concurrent test's tail requests are cancelled
    # at ~30s.
    # ------------------------------------------------------------------
    def test_probe_a_reduced_load(self):
        """
        Probe A - root-cause check. Same 16-way concurrency, but only 2
        responses per request, so the throttled workload finishes well
        within the ~30s cancellation window. If this PASSES while the
        8-response concurrent test FAILS, the failure is a throughput-vs-
        time-budget effect (the limiter starves the tail past a
        server-side cancellation), not a correctness bug.
        """
        for model_name in ("ensemble_limit_4", "ensemble_limit_1"):
            with self.subTest(probe="A-reduced-load", model=model_name):
                _dbg("PROBE-A", f"reduced load: {NUM_REQUESTS} x 2 responses, {model_name}")
                self._run_inference(
                    model_name=model_name,
                    expected_responses_per_request=2,
                    num_concurrent_requests=NUM_REQUESTS,
                )
                _dbg("PROBE-A", f"{model_name}: all {NUM_REQUESTS} requests completed")

    def test_probe_b_explicit_stream_timeout(self):
        """
        Probe B - cancellation-source check. Full 16x8 load, but with an
        explicit large client stream_timeout (120s). The client default is
        None (no timeout), so if the ~30s CANCELLED still fires WITH a 120s
        stream timeout set, the cancellation is server/transport-side, not
        the client's stream timeout.
        """
        with self.subTest(probe="B-stream-timeout-120", model="ensemble_limit_4"):
            _dbg("PROBE-B", "full load 16 x 8 with explicit stream_timeout=120s")
            self._run_inference(
                model_name="ensemble_limit_4",
                expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                num_concurrent_requests=NUM_REQUESTS,
                stream_timeout=120,
            )
            _dbg("PROBE-B", "completed without ~30s cancel (would mean client stream timeout)")

    def test_probe_c_own_connection(self):
        """
        Mechanism check: full 16x8 load, but every client gets its OWN TCP
        connection (defeat gRPC subchannel pooling). If all 16 now succeed, the
        ~30s CANCELLED was caused by too many long-lived streams sharing pooled
        connection(s) -- confirming the root cause is client-side transport, not
        the ensemble limiter. If it still cancels, the cause is a per-call client
        transport deadline instead.
        """
        with self.subTest(probe="C-own-connection", model="ensemble_limit_4"):
            _dbg("PROBE-C", "full load 16 x 8 with per-client connections (no subchannel pooling)")
            self._run_inference(
                model_name="ensemble_limit_4",
                expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                num_concurrent_requests=NUM_REQUESTS,
                channel_args=[("grpc.use_local_subchannel_pool", 1)],
            )
            _dbg("PROBE-C", "all 16 completed -> pooling/transport was the cause")

    def test_single_request_with_different_limits(self):
        """
        Single streaming request that produces 16 responses via a three-step ensemble pipeline
        (decoupled_producer → consumer_high_delay → consumer_low_delay) under various
        max_inflight_requests configurations.
        """
        cases = [
            ("ensemble_limit_4", "max_inflight_requests=4"),
            ("ensemble_limit_1", "max_inflight_requests=1"),
            ("ensemble_disabled", "max_inflight_requests is disabled"),
        ]
        for model_name, desc in cases:
            with self.subTest(limit=desc):
                self._run_inference(
                    model_name=model_name, expected_responses_per_request=16
                )

    def test_concurrent_requests_with_different_limits(self):
        """
        NUM_REQUESTS concurrent streaming requests (NUM_RESPONSES_PER_REQUEST
        responses each) exercise the max_inflight_requests limit.
        Subtests cover: limit=4, limit=1, and the limit disabled.
        """
        cases = [
            ("ensemble_limit_4", "max_inflight_requests=4"),
            ("ensemble_limit_1", "max_inflight_requests=1"),
            ("ensemble_disabled", "max_inflight_requests is disabled"),
        ]
        for model_name, desc in cases:
            with self.subTest(limit=desc):
                self._run_inference(
                    model_name=model_name,
                    expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                    num_concurrent_requests=NUM_REQUESTS,
                )

    def test_sequential_requests_limiter_resets_cleanly(self):
        """
        Send NUM_REQUESTS requests one after another. If the limiter
        leaks a slot on any request, subsequent requests will be stuck or time out.
        """
        for seq_idx in range(NUM_REQUESTS):
            with self.subTest(request=seq_idx):
                self._run_inference(
                    model_name="ensemble_limit_4",
                    expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                )

    def test_request_cancellation_under_backpressure(self):
        """
        Start a long-running request (32 responses), cancel mid-stream,
        and verify the server sends a CANCELLED status and only a partial set of
        responses is received.
        """
        input_value = 32
        user_data = UserData()

        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            inputs, outputs = prepare_infer_args(input_value, True)
            triton_client.start_stream(callback=partial(callback, user_data))

            # Start the request
            triton_client.async_stream_infer(
                model_name="ensemble_limit_4", inputs=inputs, outputs=outputs
            )

            responses = []
            try:
                result = user_data._response_queue.get(timeout=5)
                if isinstance(result, InferenceServerException):
                    self.fail(f"Got error before cancellation: {result}")
                resp = result.get_response()
                if len(resp.outputs) > 0:
                    responses.append(result)
            except queue.Empty:
                self.fail("Stream did not produce any response before cancellation.")

            # Cancel the stream - this unblocks any waiting producers and triggers a CANCELLED error.
            triton_client.stop_stream(cancel_requests=True)

            # Allow some time for cancellation
            time.sleep(1)

            cancellation_found = False
            while True:
                try:
                    result = user_data._response_queue.get(timeout=1)
                    if isinstance(result, InferenceServerException):
                        self.assertEqual(
                            result.status(),
                            "StatusCode.CANCELLED",
                            f"Expected CANCELLED status, got: {result.status()}",
                        )
                        cancellation_found = True
                        break
                    else:
                        response = result.get_response()
                        if len(response.outputs) > 0:
                            responses.append(result)
                        # Check for final response
                        final = response.parameters.get("triton_final_response")
                        if final and final.bool_param:
                            break
                except queue.Empty:
                    break

            # Verify the cancellation error was received
            self.assertTrue(
                cancellation_found,
                "Did not receive the expected cancellation error from the server.",
            )

            # Verify we received only a partial set of responses
            self.assertLess(
                len(responses),
                input_value,
                "Expected partial responses due to cancellation, but received all of them.",
            )
            self.assertGreater(
                len(responses),
                0,
                "Expected to receive at least one response before cancellation.",
            )


class EnsembleStepMaxQueueSizeTest(tu.TestResultCollector):
    def _run_inference(self, model_name, expected_responses_count):
        """
        Helper function for streaming inference.

        For decoupled streaming ensembles with queue limit on internal step:
        - Each producer response creates an independent flow through the ensemble
        - Flows that complete before error is set send their outputs successfully
        - Once error occurs (queue full), stream terminates with error
        - Result: 0-N successful responses + 1 error (N depends on timing)
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect and verify responses
                errors, responses = collect_responses(user_data)
                self.assertGreaterEqual(
                    len(responses),
                    0,
                    "May have 0 or more successful responses depending on timing",
                )
                self.assertLess(
                    len(responses),
                    expected_responses_count,
                    f"Should have fewer than {expected_responses_count} responses (some flows failed)",
                )
                self.assertEqual(
                    len(errors),
                    1,
                    "Expected exactly one error when the queue is full and the stream terminates",
                )

                # Verify correctness of successful responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertAlmostEqual(
                        output[0],
                        EXPECTED_INFER_OUTPUT,
                        places=5,
                        msg=f"Response {idx} has incorrect value - {output[0]}",
                    )

                # Verify error is queue-full error
                self.assertIn(
                    "Exceeds maximum queue size",
                    str(errors[0]),
                    f"Expected queue size error, got: {str(errors[0])}",
                )
            finally:
                triton_client.stop_stream()

    def _run_concurrent_inference(self, model_name, expected_responses_count):
        """
        Helper function for concurrent independent requests.
        Each request either succeeds completely or fails completely.
        Returns: (num_successes, num_errors) tuple
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect responses
                errors, responses = collect_responses(user_data)

                # For concurrent independent requests with queue limit on internal step:
                # - Requests that arrive before queue fills: succeed with all outputs
                # - Requests that arrive after queue fills: fail with error
                total = len(responses) + len(errors)
                self.assertEqual(
                    total,
                    expected_responses_count,
                    f"Expected {expected_responses_count} total responses, got {total}",
                )

                if len(errors) > 0:
                    # This request failed
                    self.assertEqual(
                        len(responses),
                        0,
                        "Failed request should have no successful outputs",
                    )
                    self.assertEqual(
                        len(errors), 1, "Failed request should have exactly one error"
                    )
                    self.assertIn(
                        "Exceeds maximum queue size",
                        str(errors[0]),
                        f"Expected queue size error, got: {str(errors[0])}",
                    )
                    return (0, 1)  # 0 successes, 1 error
                else:
                    # This request succeeded
                    self.assertEqual(
                        len(responses),
                        expected_responses_count,
                        f"Successful request should have all {expected_responses_count} outputs",
                    )
                    # Verify correctness of successful responses
                    for idx, resp in enumerate(responses):
                        output = resp.as_numpy("OUT")
                        self.assertAlmostEqual(
                            output[0],
                            EXPECTED_INFER_OUTPUT,
                            places=5,
                            msg=f"Response {idx} has incorrect value - {output[0]}",
                        )
                    return (expected_responses_count, 0)  # N successes, 0 errors
            finally:
                triton_client.stop_stream()

    def test_step1_max_queue_size(self):
        """
        Test max_queue_size on step 1 (decoupled_producer).

        Trigger 32 concurrent ensemble requests, each producing 1 response
        - Step 1 (producer) has max_queue_size limit
        - Some ensemble requests succeed completely (before queue fills)
        - Some fail completely (when producer queue is full)
        """
        model_name = "ensemble_step1_enabled_max_queue_size"
        num_requests = 32

        # Store results from each thread
        results = []

        def thread_wrapper(model_name, expected_count, results_list):
            """Wrapper to capture thread results"""
            result = self._run_concurrent_inference(model_name, expected_count)
            results_list.append(result)

        # Launch concurrent threads to perform infer requests
        threads = []
        for i in range(num_requests):
            t = threading.Thread(target=thread_wrapper, args=(model_name, 1, results))
            threads.append(t)
            t.start()

        # Wait for all requests to complete
        for t in threads:
            t.join(timeout=60)

        # Aggregate results from all threads
        total_successes = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)

        # Verify aggregate behavior
        self.assertEqual(
            total_successes + total_errors,
            num_requests,
            f"Expected {num_requests} total results (successes + errors), "
            f"got {total_successes} successes + {total_errors} errors = {total_successes + total_errors}",
        )

        # Verify at least some errors occurred (queue limit was hit)
        self.assertGreater(
            total_errors,
            0,
            f"Expected some errors due to max_queue_size limit, "
            f"but all {num_requests} requests succeeded.",
        )

        # Verify at least some successes occurred (not all rejected)
        self.assertGreater(
            total_successes,
            0,
            f"Expected some successful requests before queue filled, "
            f"but all {num_requests} requests failed.",
        )

    def test_step2_max_queue_size(self):
        """
        Test max_queue_size on step 2 (slow_consumer).

        Trigger 1 streaming ensemble request producing 32 responses
        - Step 1 (producer) generates 32 responses rapidly (every 100ms)
        - Step 2 (consumer) has max_queue_size=5 and processes slowly (500ms each)
        - Each producer response is an independent request to the second step through
        - the ensemble flow. Some requests complete successfully before queue fills
        - When queue fills, error is set and stream terminates
        - All inflight steps drain, then error response sent to client
        """
        model_name = "ensemble_step2_enabled_max_queue_size"
        self._run_inference(model_name=model_name, expected_responses_count=32)


class EnsembleParallelFailedEnqueueTest(tu.TestResultCollector):
    def _run_inference(self, expected_responses_count=32):
        """
        Exercise a fan-out ensemble where one parallel branch hits queue-full
        first. Successful responses emitted before the failure should still be
        correct, and the stream should terminate with exactly one queue-full
        error.
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=MODEL_ENSEMBLE_PARALLEL_FAILED_ENQUEUE,
                    inputs=inputs,
                    outputs=outputs,
                )

                errors, responses = collect_responses(user_data, timeout=15)
                self.assertLess(
                    len(responses),
                    expected_responses_count,
                    "Expected the parallel slow branch to queue-fill before all "
                    "responses completed.",
                )
                self.assertEqual(
                    len(errors),
                    1,
                    "Expected exactly one queue-full error from the parallel "
                    "failed-enqueue path.",
                )
                self.assertIn(
                    "Exceeds maximum queue size",
                    str(errors[0]),
                    f"Expected queue size error, got: {str(errors[0])}",
                )

                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertAlmostEqual(
                        float(np.squeeze(output)),
                        EXPECTED_PARALLEL_FAILED_ENQUEUE_OUTPUT,
                        places=5,
                        msg=f"Response {idx} has incorrect value - {output}",
                    )
            finally:
                triton_client.stop_stream()

    def test_parallel_step_failed_enqueue(self):
        """
        Repeat the same request according to PARALLEL_FAILED_ENQUEUE_LOOPS.
        """
        loop_count = int(os.environ.get("PARALLEL_FAILED_ENQUEUE_LOOPS", "1"))
        self.assertGreaterEqual(
            loop_count, 1, "PARALLEL_FAILED_ENQUEUE_LOOPS must be >= 1"
        )

        for iteration in range(loop_count):
            with self.subTest(iteration=iteration):
                self._run_inference()


if __name__ == "__main__":
    unittest.main()
