#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nightly_email_helper

import glob
from datetime import date

if __name__ == '__main__':
    today = date.today().strftime("%Y-%m-%d")
    subject = "Triton Server Memory Growth " + sys.argv[1] + " Summary: " + today
    memory_graphs_resnet = glob.glob("memory_growth_resnet*.log")
    memory_graphs_busyop = glob.glob("memory_growth_busyop.log")
    write_up = "<p>This test uses perf_analyzer as clients running on 4 different models. The max allowed difference between mean and maximum memory usage is set to 150MB.</p>"
    write_up += "<p><b>&#8226 What to look for</b><br>A linear memory growth in the beginning of the graph is acceptable only when it is followed by a flat memory usage. If a linear memory growth is observed during the entire test then there is possibly a memory leak.</p>"
    html_content = "<html><head></head><body><pre style=\"font-size:11pt;font-family:Arial, sans-serif;\">" + write_up + "</pre><pre style=\"font-size:11pt;font-family:Consolas;\">"
    for mem_graph in sorted(memory_graphs_resnet):
        html_content += "\n" + mem_graph + "\n"
        with open(mem_graph, "r") as f:
            html_content += f.read() + "\n"

    html_content += "<p>The busyop test is by design to show that actual memory growth is correctly detected and displayed.</p>"

    # When we see PTX failures in CI, the busyop memory graph is not created.
    if len(memory_graphs_busyop):
        write_up = "<p><b>&#8226 What to look for</b><br>The memory usage should increase continually over time, and a linear growth should be observed in the graph below.</p>"
        html_content += "</pre><pre style=\"font-size:11pt;font-family:Arial, sans-serif;\">" + write_up + "</pre><pre style=\"font-size:11pt;font-family:Consolas;\">"
        for mem_graph in sorted(memory_graphs_busyop):
            html_content += "\n" + mem_graph + "\n"
            with open(mem_graph, "r") as f:
                html_content += f.read() + "\n"
    else:
        html_content += "<p>The busyop model caused PTX failures when running the CI.</p>"
    html_content += "</pre></body></html>"
    nightly_email_helper.send(subject, html_content, is_html=True)
