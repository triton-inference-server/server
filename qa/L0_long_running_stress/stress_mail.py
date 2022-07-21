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

import os
import nightly_email_helper

from datetime import date

CI_JOB_ID = os.environ.get('CI_JOB_ID', '')

if __name__ == '__main__':
    today = date.today().strftime("%Y-%m-%d")
    subject = "Triton Long-Running Stress Test " + \
        ((sys.argv[1] + " ") if len(sys.argv) >= 2 else "") + "Summary: " + today
    stress_report = "stress_report.txt"
    link = "https://gitlab-master.nvidia.com/dl/dgx/tritonserver/-/jobs/" + CI_JOB_ID
    write_up = "<p>The table below includes results from long-running stress test. Please refer to the description of each test case to see what different kinds of inference requests were sent. Request concurrency is set to 8.</p>"
    write_up += "<p>Please check the CI output webpage for the details of the failures: " + link + "</p>"
    html_content = "<html><head></head><body><pre style=\"font-size:11pt;font-family:Arial, sans-serif;\">" + write_up + "</pre><pre style=\"font-size:11pt;font-family:Consolas;\">"
    with open(stress_report, "r") as f:
        html_content += f.read() + "\n"
    html_content += "</pre></body></html>"
    nightly_email_helper.send(subject, html_content, is_html=True)
