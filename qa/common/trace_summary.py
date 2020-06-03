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
import sys

FLAGS = None


def add_span(span_map, timestamps, span_name, ts_start, ts_end):
    for tag in (ts_start, ts_end):
        if tag not in timestamps:
            raise ValueError('timestamps missing "{}": {}'.format(
                tag, timestamps))
    if timestamps[ts_end] < timestamps[ts_start]:
        raise ValueError('end timestamp "{}" < start timestamp "{}"'.format(
            ts_end, ts_start))
    if span_name not in span_map:
        span_map[span_name] = 0
    span_map[span_name] += timestamps[ts_end] - timestamps[ts_start]


class AbstractFrontend():
    @property
    def filter_timestamp(self):
        return None

    def add_frontend_span(self, span_map, timestamps):
        pass

    def summarize_frontend_span(self, span_map, cnt):
        return None


class HttpFrontend(AbstractFrontend):
    @property
    def filter_timestamp(self):
        return "HTTP_RECV_START"

    def add_frontend_span(self, span_map, timestamps):
        if ("HTTP_RECV_START" in timestamps) and (
                "HTTP_SEND_END" in timestamps):
            add_span(span_map, timestamps, "HTTP_INFER", "HTTP_RECV_START",
                     "HTTP_SEND_END")
            add_span(span_map, timestamps, "HTTP_RECV", "HTTP_RECV_START",
                     "HTTP_RECV_END")
            add_span(span_map, timestamps, "HTTP_SEND", "HTTP_SEND_START",
                     "HTTP_SEND_END")

    def summarize_frontend_span(self, span_map, cnt):
        if "HTTP_INFER" in span_map:
            res = "HTTP infer request (avg): {}us\n".format(
                span_map["HTTP_INFER"] / (cnt * 1000))
            res += "\tReceive (avg): {}us\n".format(span_map["HTTP_RECV"] /
                                                    (cnt * 1000))
            res += "\tSend (avg): {}us\n".format(span_map["HTTP_SEND"] /
                                                 (cnt * 1000))
            res += "\tOverhead (avg): {}us\n".format(
                (span_map["HTTP_INFER"] - span_map["HTTP_RECV"] -
                 span_map["HTTP_SEND"]) / (cnt * 1000))
            return res
        else:
            return None


class GrpcFrontend(AbstractFrontend):
    @property
    def filter_timestamp(self):
        return "GRPC_WAITREAD_START"

    def add_frontend_span(self, span_map, timestamps):
        if ("GRPC_WAITREAD_START" in timestamps) and (
                "GRPC_SEND_END" in timestamps):
            add_span(span_map, timestamps, "GRPC_INFER", "GRPC_WAITREAD_START",
                     "GRPC_SEND_END")
            add_span(span_map, timestamps, "GRPC_WAITREAD",
                     "GRPC_WAITREAD_START", "GRPC_WAITREAD_END")
            add_span(span_map, timestamps, "GRPC_SEND", "GRPC_SEND_START",
                     "GRPC_SEND_END")

    def summarize_frontend_span(self, span_map, cnt):
        if "GRPC_INFER" in span_map:
            res = "GRPC infer request (avg): {}us\n".format(
                span_map["GRPC_INFER"] / (cnt * 1000))
            res += "\tWait/Read (avg): {}us\n".format(
                span_map["GRPC_WAITREAD"] / (cnt * 1000))
            res += "\tSend (avg): {}us\n".format(span_map["GRPC_SEND"] /
                                                 (cnt * 1000))
            res += "\tOverhead (avg): {}us\n".format(
                (span_map["GRPC_INFER"] - span_map["GRPC_WAITREAD"] -
                 span_map["GRPC_SEND"]) / (cnt * 1000))
            return res
        else:
            return None


def summarize(frontend, traces):
    # map from (model_name, model_version) to # of traces
    model_count_map = dict()
    # map from (model_name, model_version) to map of span->total time
    model_span_map = dict()

    # Order traces by id to be more intuitive if 'show_trace'
    traces = sorted(traces, key=lambda t: t.get('id', -1))

    # Filter the trace that is not for the requested frontend
    match_frontend_id_set = set()
    # Filter the trace that is not meaningful and group them by 'id'
    filtered_traces = dict()
    for trace in traces:
        if "id" not in trace:
            continue

        # Trace without a parent must contain frontend timestamps
        add_trace = False
        if "parent_id" not in trace:
            if frontend.filter_timestamp is None:
                continue
            if "timestamps" in trace:
                for ts in trace["timestamps"]:
                    if frontend.filter_timestamp in ts["name"]:
                        match_frontend_id_set.add(trace["id"])
            if trace["id"] in match_frontend_id_set:
                add_trace = True
        # Otherwise need to check whether parent is filtered
        elif trace["parent_id"] in match_frontend_id_set:
            match_frontend_id_set.add(trace["id"])
            add_trace = True

        if add_trace:
            if (trace['id'] in filtered_traces.keys()):
                rep_trace = filtered_traces[trace['id']]
                # Apend the timestamp to the trace representing this 'id'
                if "model_name" in trace:
                    rep_trace["model_name"] = trace["model_name"]
                if "model_version" in trace:
                    rep_trace["model_version"] = trace["model_version"]
                if "timestamps" in trace:
                    rep_trace["timestamps"] += trace["timestamps"]
            else:
                # Use this trace to represent this 'id'
                if "timestamps" not in trace:
                    trace["timestamps"] = []
                filtered_traces[trace['id']] = trace

    for trace_id, trace in filtered_traces.items():
        if trace_id not in match_frontend_id_set:
            filtered_traces.pop(trace_id, None)
            continue
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]
        if ("REQUEST_START" in timestamps) and ("REQUEST_END" in timestamps):
            key = (trace["model_name"], trace["model_version"])
            if key not in model_count_map:
                model_count_map[key] = 0
                model_span_map[key] = dict()

            model_count_map[key] += 1

            frontend.add_frontend_span(model_span_map[key], timestamps)

            add_span(model_span_map[key], timestamps, "REQUEST",
                     "REQUEST_START", "REQUEST_END")

            # The tags below will be missing for ensemble model
            if ("QUEUE_START" in timestamps) and (
                    "COMPUTE_START" in timestamps):
                add_span(model_span_map[key], timestamps, "QUEUE",
                         "QUEUE_START", "COMPUTE_START")
            if ("COMPUTE_START" in timestamps) and (
                    "COMPUTE_END" in timestamps):
                add_span(model_span_map[key], timestamps, "COMPUTE",
                         "COMPUTE_START", "COMPUTE_END")
            if ("COMPUTE_INPUT_END" in timestamps) and (
                    "COMPUTE_OUTPUT_START" in timestamps):
                add_span(model_span_map[key], timestamps, "COMPUTE_INPUT",
                         "COMPUTE_START", "COMPUTE_INPUT_END")
                add_span(model_span_map[key], timestamps, "COMPUTE_INFER",
                         "COMPUTE_INPUT_END", "COMPUTE_OUTPUT_START")
                add_span(model_span_map[key], timestamps, "COMPUTE_OUTPUT",
                         "COMPUTE_OUTPUT_START", "COMPUTE_END")
            if FLAGS.show_trace:
                print("{} ({}):".format(trace["model_name"],
                                        trace["model_version"]))
                print("\tid: {}".format(trace["id"]))
                if "parent_id" in trace:
                    print("\tparent id: {}".format(trace["parent_id"]))
                ordered_timestamps = list()
                for ts in trace["timestamps"]:
                    ordered_timestamps.append((ts["name"], ts["ns"]))
                ordered_timestamps.sort(key=lambda tup: tup[1])

                now = None
                for ts in ordered_timestamps:
                    if now is not None:
                        print("\t\t{}us".format((ts[1] - now) / 1000))
                    print("\t{}".format(ts[0]))
                    now = ts[1]

    for key, cnt in model_count_map.items():
        model_name, model_value = key
        print("Summary for {} ({}): trace count = {}".format(
            model_name, model_value, cnt))

        frontend_summary = frontend.summarize_frontend_span(
            model_span_map[key], cnt)
        if frontend_summary is not None:
            print(frontend_summary)

        print("\tHandler (avg): {}us".format(model_span_map[key]["REQUEST"] /
                                             (cnt * 1000)))
        if ("QUEUE" in model_span_map[key]
           ) and "COMPUTE" in model_span_map[key]:
            print("\t\tOverhead (avg): {}us".format(
                (model_span_map[key]["REQUEST"] - model_span_map[key]["QUEUE"] -
                 model_span_map[key]["COMPUTE"]) / (cnt * 1000)))
            print("\t\tQueue (avg): {}us".format(model_span_map[key]["QUEUE"] /
                                                 (cnt * 1000)))
            print("\t\tCompute (avg): {}us".format(
                model_span_map[key]["COMPUTE"] / (cnt * 1000)))
        if ("COMPUTE_INPUT" in model_span_map[key]
           ) and "COMPUTE_OUTPUT" in model_span_map[key]:
            print("\t\t\tInput (avg): {}us".format(
                model_span_map[key]["COMPUTE_INPUT"] / (cnt * 1000)))
            print("\t\t\tInfer (avg): {}us".format(
                model_span_map[key]["COMPUTE_INFER"] / (cnt * 1000)))
            print("\t\t\tOutput (avg): {}us".format(
                model_span_map[key]["COMPUTE_OUTPUT"] / (cnt * 1000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-t',
                        '--show-trace',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Show timestamps for each individual trace')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    FLAGS = parser.parse_args()

    for f in FLAGS.file:
        trace_data = json.loads(f.read())
        if FLAGS.verbose:
            print(json.dumps(trace_data, sort_keys=True, indent=2))

        # Must summarize HTTP and GRPC separately since they have
        # different ways of accumulating time.
        print("File: {}".format(f.name))
        summarize(HttpFrontend(), trace_data)
        summarize(GrpcFrontend(), trace_data)
