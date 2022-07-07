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
    subject = "Triton Client Memory Growth " + sys.argv[1] + " Summary: " + today
    memory_graphs = glob.glob("client_memory_growth*.log")
    write_up = "<p>This test is run for both HTTP and GRPC protocols using C++ and Python test scripts. The max-allowed difference between mean and maximum memory usage is set to 10MB and 1MB for C++ and Python tests individually.</p>"
    write_up += "<p><b>&#8226 What to look for</b><br>A linear memory growth in the beginning of the graph is acceptable only when it is followed by a flat memory usage. If a linear memory growth is observed during the entire test then there is possibly a memory leak.</p>"
    html_content = "<html><head></head><body><pre style=\"font-size:11pt;font-family:Arial, sans-serif;\">" + write_up + "</pre><pre style=\"font-size:11pt;font-family:Consolas;\">"
    for mem_graph in sorted(memory_graphs):
        html_content += "\n" + mem_graph + "\n"
        with open(mem_graph, "r") as f:
            html_content += f.read() + "\n"
    html_content += "</pre></body></html>"
    nightly_email_helper.send(subject, html_content, is_html=True)
