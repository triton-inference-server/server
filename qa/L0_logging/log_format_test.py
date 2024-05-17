# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import google.protobuf.text_format
import numpy
import pytest
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

module_directory = os.path.split(os.path.abspath(__file__))[0]

test_model_directory = os.path.abspath(os.path.join(module_directory, "log_models"))


test_logs_directory = os.path.abspath(
    os.path.join(module_directory, "log_format_test_logs")
)

shutil.rmtree(test_logs_directory, ignore_errors=True)

os.makedirs(test_logs_directory)

# Regular expressions for Table
#
# Table format is:
#
# border
# header_row
# border
# data_rows
# border

table_border_regex = re.compile(r"^\+[-+]+\+$")
table_row_regex = re.compile(r"^\| (?P<row>.*?) \|$")


# Regular expression pattern for default log record
DEFAULT_LOG_RECORD = r"(?P<level>\w)(?P<month>\d{2})(?P<day>\d{2}) (?P<timestamp>\d{2}:\d{2}:\d{2}\.\d{6}) (?P<pid>\d+) (?P<file>[\w\.]+):(?P<line>\d+)] (?P<message>.*)"
default_log_record_regex = re.compile(DEFAULT_LOG_RECORD, re.DOTALL)

# Regular expression pattern for ISO8601 log record
ISO8601_LOG_RECORD = r"(?P<ISO8601_timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) (?P<level>\w+) (?P<pid>\d+) (?P<file>.+):(?P<line>\d+)] (?P<message>.*)"
IS08601_log_record_regex = re.compile(ISO8601_LOG_RECORD, re.DOTALL)

LEVELS = set({"E", "W", "I"})

FORMATS = [
    ("default", default_log_record_regex),
    ("ISO8601", IS08601_log_record_regex),
    ("default_unescaped", default_log_record_regex),
    ("ISO8601_unescaped", IS08601_log_record_regex),
]

IDS = ["default", "ISO8601", "default_unescaped", "ISO8601_unescaped"]


def parse_timestamp(timestamp):
    hours, minutes, seconds = timestamp.split(":")
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)
    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)


validators = {}


def validator(func):
    validators[func.__name__.replace("validate_", "")] = func
    return func


@validator
def validate_level(level, _):
    assert level in LEVELS


@validator
def validate_month(month, _):
    assert month.isdigit()
    month = int(month)
    assert month >= 1 and month <= 12


@validator
def validate_day(day, _):
    assert day.isdigit()
    day = int(day)
    assert day >= 1 and day <= 31


@validator
def validate_ISO8601_timestamp(timestamp, _):
    datetime.datetime.fromisoformat(timestamp.rstrip("Z"))


@validator
def validate_timestamp(timestamp, _):
    parse_timestamp(timestamp)


@validator
def validate_pid(pid, _):
    assert pid.isdigit()


@validator
def validate_file(file_, _):
    assert Path(file_).name is not None


@validator
def validate_line(line, _):
    assert line.isdigit()


def _split_row(row):
    return [r.strip() for r in row.group("row").strip().split("|")]


def validate_table(table_rows):
    index = 0
    top_border = table_border_regex.search(table_rows[index])
    assert top_border

    index += 1
    header = table_row_regex.search(table_rows[index])
    assert header
    header = _split_row(header)

    index += 1
    middle_border = table_border_regex.search(table_rows[index])
    assert middle_border

    # Process each row
    index += 1
    parsed_rows = []
    row = ""
    for index, row in enumerate(table_rows[index:]):
        matched = table_row_regex.search(row)
        if matched:
            row_data = _split_row(matched)
            parsed_rows.append(row_data)

    end_border = table_border_regex.search(row)
    assert end_border

    for row in parsed_rows:
        assert len(row) == len(header)


@validator
def validate_message(message, escaped):
    heading, obj = message.split("\n", 1)
    if heading and escaped:
        try:
            json.loads(heading)
        except json.JSONDecodeError as e:
            raise Exception(
                f"{e} First line of message in log record is not a valid JSON string"
            )
        except Exception as e:
            raise type(e)(
                f"{e} First line of message in log record is not a valid JSON string"
            )
    if len(obj):
        obj = obj.strip().split("\n")
        if obj:
            match = table_border_regex.search(obj[0])
            if match:
                validate_table(obj)
            else:
                google.protobuf.text_format.ParseLines(
                    obj, grpcclient.model_config_pb2.ModelConfig()
                )


class TestLogFormat:
    @pytest.fixture(autouse=True)
    def setup(self, request):
        test_case_name = request.node.name
        self._server_options = {}
        self._server_options["log-verbose"] = 256
        self._server_options["log-info"] = 1
        self._server_options["log-error"] = 1
        self._server_options["log-warning"] = 1
        self._server_options["log-format"] = "default"
        self._server_options["model-repository"] = os.path.abspath(
            os.path.join(module_directory, "log_models")
        )
        self._server_process = None
        self._server_options["log-file"] = os.path.join(
            test_logs_directory, test_case_name + ".server.log"
        )

    def _launch_server(self, escaped=None):
        cmd = ["tritonserver"]

        for key, value in self._server_options.items():
            cmd.append(f"--{key}={value}")

        env = os.environ.copy()

        if escaped is not None and not escaped:
            env["TRITON_SERVER_ESCAPE_LOG_MESSSAGES"] = "FALSE"
        elif escaped is not None and escaped:
            env["TRITON_SERVER_ESCAPE_LOG_MESSSAGES"] = "TRUE"
        else:
            del env["TRITON_SERVER_ESCAPE_LOG_MESSSAGES"]

        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        wait_time = 5

        while wait_time and not os.path.exists(self._server_options["log-file"]):
            time.sleep(1)
            wait_time -= 1

        if not os.path.exists(self._server_options["log-file"]):
            raise Exception("Log not found")

    def validate_log_record(self, record, format_regex, escaped):
        match = format_regex.search(record)
        assert match, "Invalid log line"

        for field, value in match.groupdict().items():
            if field not in validators:
                continue
            try:
                validators[field](value, escaped)
            except Exception as e:
                raise type(e)(
                    f"{e}\nInvalid {field}: '{match.group(field)}' in log record '{record}'"
                )

    def verify_log_format(self, file_path, format_regex, escaped):
        log_records = []
        with open(file_path, "rt") as file_:
            current_log_record = []
            for line in file_:
                match = format_regex.search(line)
                if match:
                    if current_log_record:
                        log_records.append(current_log_record)
                    current_log_record = [line]
                else:
                    current_log_record.append(line)
        log_records.append(current_log_record)
        log_records = ["".join(log_record_lines) for log_record_lines in log_records]
        for log_record in log_records:
            self.validate_log_record(log_record, format_regex, escaped)

    @pytest.mark.parametrize(
        "log_format,format_regex",
        FORMATS,
        ids=IDS,
    )
    def test_log_format(self, log_format, format_regex):
        self._server_options["log-format"] = log_format.replace("_unescaped", "")

        escaped = "_unescaped" not in log_format

        self._launch_server(escaped)
        time.sleep(1)
        self._server_process.kill()
        self._server_process.wait()
        self.verify_log_format(self._server_options["log-file"], format_regex, escaped)

    def foo_test_injection(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit(1)

        input_name = "'nothing_wrong'\nI0205 18:34:18.707423 1 [file.cc:123] THIS ENTRY WAS INJECTED\nI0205 18:34:18.707461 1 [http_server.cc:3570] [request id: <id_unknown>] Infer failed: [request id: <id_unknown>] input 'nothing_wrong"

        input_data = numpy.random.randn(1, 3).astype(numpy.float32)
        input_tensor = httpclient.InferInput(input_name, input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        triton_client.infer(model_name="simple", inputs=[input_tensor])
