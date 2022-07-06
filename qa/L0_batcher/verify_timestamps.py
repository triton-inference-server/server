#!/usr/bin/python
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

import argparse
import json

FLAGS = None


def verify_timestamps(traces, preserve):
    # Order traces by id
    traces = sorted(traces, key=lambda t: t.get('id', -1))

    # Filter the trace that is not meaningful and group them by 'id'
    filtered_traces = dict()
    grpc_id_offset = 0
    for trace in traces:
        if "id" not in trace:
            continue
        # Skip GRPC traces as actual traces are not genarated via GRPC,
        # thus GRPC traces are ill-formed
        if "timestamps" in trace:
            is_grpc = False
            for ts in trace["timestamps"]:
                if "GRPC" in ts["name"]:
                    is_grpc = True
                    break
            if is_grpc:
                grpc_id_offset += 1
                continue

        if (trace['id'] in filtered_traces.keys()):
            rep_trace = filtered_traces[trace['id']]
            # Apend the timestamp to the trace representing this 'id'
            if "timestamps" in trace:
                rep_trace["timestamps"] += trace["timestamps"]
        else:
            # Use this trace to represent this 'id'
            if "timestamps" not in trace:
                trace["timestamps"] = []
            filtered_traces[trace['id']] = trace

    # First find the latest response complete timestamp for the batch with large delay
    large_delay_response_complete = 0
    small_delay_traces = []
    for trace_id, trace in filtered_traces.items():
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        # Hardcoded delay value here (knowing large delay is 400ms)
        compute_span = timestamps["COMPUTE_END"] - timestamps["COMPUTE_START"]
        # If the 3rd batch is also processed by large delay instance, we don't
        # want to use its responses as baseline
        if trace["id"] <= (
                8 + grpc_id_offset) and compute_span >= 400 * 1000 * 1000:
            response_complete = timestamps["INFER_RESPONSE_COMPLETE"]
            large_delay_response_complete = max(large_delay_response_complete,
                                                response_complete)
        else:
            small_delay_traces.append(trace)

    response_request_after_large_delay_count = 0
    for trace in small_delay_traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        response_complete = timestamps["INFER_RESPONSE_COMPLETE"]
        if response_complete > large_delay_response_complete:
            response_request_after_large_delay_count += 1

    # Hardcoded expected count here
    print("responses after large delay count: {}".format(
        response_request_after_large_delay_count))
    if preserve:
        # If preserve ordering, there must be large delay batch followed by
        # small delay batch and thus at least 4 responses are sent after
        return 0 if response_request_after_large_delay_count >= 4 else 1
    else:
        # If not preserve ordering, the small delay batches should all be done
        # before large delay batch regardless of the ordering in scheduler
        return 0 if response_request_after_large_delay_count == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--preserve',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Timestamps is collected with preserve ordering')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    FLAGS = parser.parse_args()

    for f in FLAGS.file:
        trace_data = json.loads(f.read())
        exit(verify_timestamps(trace_data, FLAGS.preserve))
