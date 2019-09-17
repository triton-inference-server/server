#!/usr/bin/python

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

import argparse
import json
import sys

FLAGS = None

def add_span(span_map, timestamps, span_name, ts_start, ts_end):
    if timestamps[ts_end] < timestamps[ts_start]:
        raise ValueError('end timestamp "{}" < start timestamp "{}"'.format(ts_end, ts_start))
    if span_name not in span_map:
        span_map[span_name] = 0
    span_map[span_name] += timestamps[ts_end] - timestamps[ts_start]

def summarize(traces):
    # map from (model_name, model_version) to # of traces
    model_count_map = dict()
    # map from (model_name, model_version) to map of span->total time
    model_span_map = dict()

    for trace in traces:
        timestamps = dict()
        for ts in trace["timestamps"]:
            timestamps[ts["name"]] = ts["ns"]

        key = (trace["model_name"], trace["model_version"])
        if key not in model_count_map:
            model_count_map[key] = 0
            model_span_map[key] = dict()

        model_count_map[key] += 1
        add_span(model_span_map[key], timestamps,
                 "request handler", "request handler start", "request handler end")
        add_span(model_span_map[key], timestamps,
                 "api receive", "api request start", "request handler start")
        add_span(model_span_map[key], timestamps,
                 "queue", "queue start", "compute start")
        add_span(model_span_map[key], timestamps,
                 "compute", "compute start", "compute end")
        add_span(model_span_map[key], timestamps,
                 "api send", "compute end", "api request end")
        if ("compute input end" in timestamps) and ("compute output start" in timestamps):
            add_span(model_span_map[key], timestamps,
                     "compute input", "compute start", "compute input end")
            add_span(model_span_map[key], timestamps,
                     "compute infer", "compute input end", "compute output start")
            add_span(model_span_map[key], timestamps,
                     "compute output", "compute output start", "compute end")

        if FLAGS.show_trace:
            print("{} ({}):".format(trace["model_name"], trace["model_version"]));
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
        print("Summary for {} ({}): trace count = {}".format(model_name, model_value, cnt))
        print("Server API receive (avg): {}us".format(model_span_map[key]["api receive"] / cnt / 1000))
        print("Server request (avg): {}us".format(model_span_map[key]["request handler"] / cnt / 1000))
        print("Server queue (avg): {}us".format(model_span_map[key]["queue"] / cnt / 1000))
        print("Server compute (avg): {}us".format(model_span_map[key]["compute"] / cnt / 1000))
        if ("compute input" in model_span_map[key]) and "compute output" in model_span_map[key]:
            print("\tinput (avg): {}us".format(model_span_map[key]["compute input"] / cnt / 1000))
            print("\tinfer (avg): {}us".format(model_span_map[key]["compute infer"] / cnt / 1000))
            print("\toutput (avg): {}us".format(model_span_map[key]["compute output"] / cnt / 1000))
        print("Server API send (avg): {}us".format(model_span_map[key]["api send"] / cnt / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-t', '--show-trace', action="store_true", required=False, default=False,
                        help='Show timestamps for each individual trace')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    FLAGS = parser.parse_args()

    for f in FLAGS.file:
        trace_data = json.loads(f.read())
        if FLAGS.verbose:
            print json.dumps(trace_data, sort_keys=True, indent=2)

        print("File: {}".format(f.name))
        summarize(trace_data)
