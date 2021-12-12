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

from email import encoders
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import glob
import os
import sys
import tarfile


def send(subject: str,
         content: str,
         attachments=None,
         files_to_tar=None,
         is_html=False):
    FROM = os.environ.get('TRITON_FROM', '')
    TO = os.environ.get('TRITON_TO_DL', '')
    if FROM == '' or TO == '':
        print('Must set TRITON_FROM and TRITON_TO_DL env variables')
        sys.exit(1)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = FROM
    msg['To'] = TO
    if is_html:
        mime_text = MIMEText(content, 'html')
    else:
        mime_text = MIMEText(content)
    msg.attach(mime_text)

    if attachments is None:
        attachments = []

    if files_to_tar is not None:
        with tarfile.open(subject + ".tgz", "w:gz") as csv_tar:
            for filename in glob.glob(files_to_tar):
                csv_tar.add(filename)
        attachments.append(subject + ".tgz")

    for fname in attachments:
        p = MIMEBase('application', 'octet-stream')
        with open(fname, "rb") as attachment:
            p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition',
                     "attachment; filename= %s" % (fname))
        msg.attach(p)

    mailServer = smtplib.SMTP("mailgw.nvidia.com")
    mailServer.send_message(msg)
    mailServer.quit()
