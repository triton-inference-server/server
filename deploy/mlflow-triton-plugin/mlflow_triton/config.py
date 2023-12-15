#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import re
from collections import namedtuple

from mlflow.exceptions import MlflowException


class Config(dict):
    def __init__(self):
        super().__init__()
        self["triton_url"] = os.environ.get("TRITON_URL")
        self["triton_model_repo"] = os.environ.get("TRITON_MODEL_REPO")

        if self["triton_model_repo"].startswith("s3://"):
            self.s3_regex = re.compile(
                "s3://(http://|https://|)([0-9a-zA-Z\\-.]+):([0-9]+)/"
                "([0-9a-z.\\-]+)(((/[0-9a-zA-Z.\\-_]+)*)?)"
            )

            uri = self.parse_path(self["triton_model_repo"])
            if uri.protocol == "https://":
                protocol = "https://"
            else:
                protocol = "http://"
            endpoint_url = None
            if uri.host_name != "" and uri.host_port != "":
                endpoint_url = "{}{}:{}".format(protocol, uri.host_name, uri.host_port)

            import boto3

            # boto3 handles AWS credentials
            self["s3"] = boto3.client("s3", endpoint_url=endpoint_url)
            self["s3_bucket"] = uri.bucket
            self["s3_prefix"] = uri.prefix
            self["triton_model_repo"] = "s3://{}".format(
                os.path.join(uri.bucket, uri.prefix)
            )

    def parse_path(self, path):
        # Cleanup extra slashes
        clean_path = self.clean_path(path)

        # Get the bucket name and the object path. Return error if path is malformed
        match = self.s3_regex.fullmatch(clean_path)
        S3URI = namedtuple(
            "S3URI", ["protocol", "host_name", "host_port", "bucket", "prefix"]
        )
        if match:
            uri = S3URI(*match.group(1, 2, 3, 4, 5))
            if uri.prefix and uri.prefix[0] == "/":
                uri = uri._replace(prefix=uri.prefix[1:])
        else:
            bucket_start = clean_path.find("s3://") + len("s3://")
            bucket_end = clean_path.find("/", bucket_start)

            # If there isn't a slash, the address has only the bucket
            if bucket_end > bucket_start:
                bucket = clean_path[bucket_start:bucket_end]
                prefix = clean_path[bucket_end + 1 :]
            else:
                bucket = clean_path[bucket_start:]
                prefix = ""
            uri = S3URI("", "", "", bucket, prefix)

        if uri.bucket == "":
            raise MlflowException("No bucket name found in path: " + path)

        return uri

    def clean_path(self, s3_path):
        # Must handle paths with s3 prefix
        start = s3_path.find("s3://")
        path = ""
        if start != -1:
            path = s3_path[start + len("s3://") :]
            clean_path = "s3://"
        else:
            path = s3_path
            clean_path = ""

        # Must handle paths with https:// or http:// prefix
        https_start = path.find("https://")
        if https_start != -1:
            path = path[https_start + len("https://") :]
            clean_path += "https://"
        else:
            http_start = path.find("http://")
            if http_start != -1:
                path = path[http_start + len("http://") :]
                clean_path += "http://"

        # Remove trailing slashes
        rtrim_length = len(path.rstrip("/"))
        if rtrim_length == 0:
            raise MlflowException("Invalid bucket name: '" + path + "'")

        # Remove leading slashes
        ltrim_length = len(path) - len(path.lstrip("/"))
        if ltrim_length == len(path):
            raise MlflowException("Invalid bucket name: '" + path + "'")

        # Remove extra internal slashes
        true_path = path[ltrim_length : rtrim_length + 1]
        previous_slash = False
        for i in range(len(true_path)):
            if true_path[i] == "/":
                if not previous_slash:
                    clean_path += true_path[i]
                previous_slash = True
            else:
                clean_path += true_path[i]
                previous_slash = False

        return clean_path
