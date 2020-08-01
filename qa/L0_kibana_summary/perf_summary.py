#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import summarize_kibana as sk
from email import encoders
from datetime import date
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import base64
import glob


def send_email(subject, html, suffix):
    FROM = "hemantj@nvidia.com"
    TO = 'sw-dl-triton@exchange.nvidia.com'
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = FROM
    msg['To'] = TO
    msg.attach(MIMEText(html, "html"))

    for filename in glob.glob(suffix + "*.csv"):
        attachment = open(filename, "rb")
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename)
        msg.attach(p)

    mailServer = smtplib.SMTP("mailgw.nvidia.com")
    mailServer.sendmail(FROM, TO, msg.as_string())
    mailServer.quit()


# NoModel
html = '<html><body><a href=\"https://gpuwa.nvidia.com/kibana/app/kibana#/dashboard/ff9a1030-9a1c-11ea-8edb-c5a5e5f9de0d\">Nomodel Kibana Dashboard</a><br><center>'
today = date.today().strftime("%Y-%m-%d")
for metric in ["latency", "throughput"]:
    for payload_size in ["1", "4194304"]:
        for protocol in ["grpc", "http"]:
            if metric == "throughput":
                instances = "2"
                values = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
            else:
                instances = "1"
                values = ["d_latency_p95_ms", "s_framework", "\'@timestamp\'"]
            payload_label = "4B" if payload_size == "1" else "16MB"

            where_dict = {
                "s_shared_memory": "none",
                "s_benchmark_name": "nomodel",
                "l_size": payload_size,
                "s_protocol": protocol,
                "l_instance_count": instances
            }

            title = "Nomodel " + protocol.upper() + " " + metric + " with " + \
                payload_label + " payload"
            ma_df = sk.current_moving_average_dataframe(metric,
                                                        values,
                                                        where_dict,
                                                        today,
                                                        plot=True,
                                                        plot_file=title +
                                                        ".png")
            ma_df.to_csv(title + ".csv", index=False)
            img = open(title + ".png", "rb")
            data_uri = base64.b64encode(img.read()).decode('ascii')
            html += "<img src=\"data:image/png;base64,{0}\">".format(data_uri)
            html += "<div style=\"font-weight:bold\">{0}</div>".format(title)
            html += "<br><br>"

html += '</center></body></html>'
send_email("Triton Nomodel Performance Summary: " + today, html, "Nomodel")

# Resnet50
html = '<html><body><a href=\"https://gpuwa.nvidia.com/kibana/app/kibana#/dashboard/072b6f60-b02f-11ea-9584-77098036527d\">Resnet50 Kibana Dashboard</a><br><center>'
today = date.today().strftime("%Y-%m-%d")
for batch_size in ["1", "128"]:
    for metric in ["latency", "throughput"]:
        for protocol in ["grpc", "http"]:
            instances = "2" if batch_size == "128" else "1"
            if metric == "throughput":
                values = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
            else:
                values = ["d_latency_p95_ms", "s_framework", "\'@timestamp\'"]

            where_dict = {
                "s_benchmark_name": "resnet50",
                "l_batch_size": batch_size,
                "s_protocol": protocol,
                "l_instance_count": instances
            }

            title = "Resnet50 " + protocol.upper(
            ) + " " + metric + " with Batch Size " + batch_size
            ma_df = sk.current_moving_average_dataframe(metric,
                                                        values,
                                                        where_dict,
                                                        today,
                                                        plot=True,
                                                        plot_file=title +
                                                        ".png")
            ma_df.to_csv(title + ".csv", index=False)
            img = open(title + ".png", "rb")
            data_uri = base64.b64encode(img.read()).decode('ascii')
            html += "<img src=\"data:image/png;base64,{0}\">".format(data_uri)
            html += "<div style=\"font-weight:bold\">{0}</div>".format(title)
            html += "<br><br>"

SUBJECT = "Triton Resnet50 Performance Summary: " + today
html += '</center></body></html>'
send_email("Triton Resnet50 Performance Summary: " + today, html, "Resnet50")

# Deeprecommender
html = '<html><body><a href=\"https://gpuwa.nvidia.com/kibana/app/kibana#/visualize/edit/b5bf0030-c53a-11ea-a3f8-d19d9c5c9954\">Deeprecommender Kibana Dashboard</a><br><center>'
today = date.today().strftime("%Y-%m-%d")
for batch_size in ["1", "256"]:
    for metric in ["latency", "throughput"]:
        for protocol in ["grpc", "http"]:
            instances = "2" if batch_size == "256" else "1"
            if metric == "throughput":
                values = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
            else:
                values = ["d_latency_p95_ms", "s_framework", "\'@timestamp\'"]

            where_dict = {
                "s_benchmark_name": "deeprecommender",
                "l_batch_size": batch_size,
                "s_protocol": protocol,
                "l_instance_count": instances
            }

            title = "Deeprecommender " + protocol.upper(
            ) + " " + metric + " with Batch Size " + batch_size
            ma_df = sk.current_moving_average_dataframe(metric,
                                                        values,
                                                        where_dict,
                                                        today,
                                                        plot=True,
                                                        plot_file=title +
                                                        ".png")
            ma_df.to_csv(title + ".csv", index=False)
            img = open(title + ".png", "rb")
            data_uri = base64.b64encode(img.read()).decode('ascii')
            html += "<img src=\"data:image/png;base64,{0}\">".format(data_uri)
            html += "<div style=\"font-weight:bold\">{0}</div>".format(title)
            html += "<br><br>"

html += '</center></body></html>'
send_email("Triton Deeprecommender Performance Summary: " + today, html,
           "Deeprecommender")

# Kaldi
html = '<html><body><a href=\"https://gpuwa.nvidia.com/kibana/app/kibana#/dashboard/a2a27bb0-b4b1-11ea-9584-77098036527d\">Kaldi Kibana Dashboard</a><br><center>'
today = date.today().strftime("%Y-%m-%d")
for metric in ["latency", "throughput"]:
    protocol = "grpc"
    if metric == "throughput":
        values = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
    else:
        values = ["d_latency_p95_ms", "s_framework", "\'@timestamp\'"]

    where_dict = {
        "s_benchmark_name": "kaldi",
        "s_model": "asr_kaldi",
        "l_instance_count": "1"
    }

    title = "ASR Kaldi " + protocol.upper(
    ) + " " + metric + " with concurrency 2000"
    ma_df = sk.current_moving_average_dataframe(metric,
                                                values,
                                                where_dict,
                                                today,
                                                plot=True,
                                                plot_file=title + ".png")
    ma_df.to_csv(title + ".csv", index=False)
    img = open(title + ".png", "rb")
    data_uri = base64.b64encode(img.read()).decode('ascii')
    html += "<img src=\"data:image/png;base64,{0}\">".format(data_uri)
    html += "<div style=\"font-weight:bold\">{0}</div>".format(title)
    html += "<br><br>"

html += '</center></body></html>'
send_email("Triton Kaldi Performance Summary: " + today, html, "ASR Kaldi")
