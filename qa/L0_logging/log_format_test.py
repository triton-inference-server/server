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
import time
from pathlib import Path

import google.protobuf.text_format
import numpy
import pytest
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

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
ISO8601_log_record_regex = re.compile(ISO8601_LOG_RECORD, re.DOTALL)

LEVELS = set({"E", "W", "I"})

FORMATS = [
    ("default", default_log_record_regex),
    ("ISO8601", ISO8601_log_record_regex),
    ("default_unescaped", default_log_record_regex),
    ("ISO8601_unescaped", ISO8601_log_record_regex),
]

IDS = ["default", "ISO8601", "default_unescaped", "ISO8601_unescaped"]

INT32_MAX = 2**31 - 1

INJECTED_MESSAGE = "THIS ENTRY WAS INJECTED"

CONTROL_INJECTED_MESSAGE = (
    "\u001b[31mESC-INJECTION-LFUNICODE:\u001b[32mSUCCESSFUL\u001b[0m\u0007"
)

DEFAULT_INJECTED_LOG_FORMAT = (
    "I0205 18:34:18.707423 1 file.cc:123] {QUOTE}{INJECTED_MESSAGE}{QUOTE}"
)
ISO8601_INJECTED_LOG_FORMAT = (
    "2024-05-18T01:46:51Z I 1 file.cc:123] {QUOTE}{INJECTED_MESSAGE}{QUOTE}"
)

INJECTED_FORMATS = [
    (
        "default",
        default_log_record_regex,
        DEFAULT_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=INJECTED_MESSAGE, QUOTE='"'
        ),
    ),
    (
        "ISO8601",
        ISO8601_log_record_regex,
        ISO8601_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=INJECTED_MESSAGE, QUOTE='"'
        ),
    ),
    (
        "default_unescaped",
        default_log_record_regex,
        DEFAULT_INJECTED_LOG_FORMAT.format(INJECTED_MESSAGE=INJECTED_MESSAGE, QUOTE=""),
    ),
    (
        "ISO8601_unescaped",
        ISO8601_log_record_regex,
        ISO8601_INJECTED_LOG_FORMAT.format(INJECTED_MESSAGE=INJECTED_MESSAGE, QUOTE=""),
    ),
    (
        "default",
        default_log_record_regex,
        DEFAULT_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=CONTROL_INJECTED_MESSAGE, QUOTE='"'
        ),
    ),
    (
        "ISO8601",
        ISO8601_log_record_regex,
        ISO8601_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=CONTROL_INJECTED_MESSAGE, QUOTE='"'
        ),
    ),
    (
        "default_unescaped",
        default_log_record_regex,
        DEFAULT_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=CONTROL_INJECTED_MESSAGE, QUOTE=""
        ),
    ),
    (
        "ISO8601_unescaped",
        ISO8601_log_record_regex,
        ISO8601_INJECTED_LOG_FORMAT.format(
            INJECTED_MESSAGE=CONTROL_INJECTED_MESSAGE, QUOTE=""
        ),
    ),
]

INJECTED_IDS = [
    "default",
    "ISO8601",
    "default_unescaped",
    "ISO8601_unescaped",
    "default_control",
    "ISO8601_control",
    "default_unescaped_control",
    "ISO8601_unescaped_control",
]

ESCAPE_ENVIRONMENT_VARIABLE = "TRITON_SERVER_ESCAPE_LOG_MESSAGES"


class LogInjectionError(Exception):
    pass


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


def split_row(row):
    return [r.strip() for r in row.group("row").strip().split("|")]


def validate_protobuf(protobuf):
    # Note currently we only check for model config
    # but technically any protubuf should be valid

    google.protobuf.text_format.ParseLines(
        protobuf, grpcclient.model_config_pb2.ModelConfig()
    )


def validate_table(table_rows):
    index = 0
    top_border = table_border_regex.search(table_rows[index])
    assert top_border

    index += 1
    header = table_row_regex.search(table_rows[index])
    assert header
    header = split_row(header)

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
            row_data = split_row(matched)
            parsed_rows.append(row_data)

    end_border = table_border_regex.search(row)
    assert end_border

    for row in parsed_rows:
        assert len(row) == len(header)


@validator
def validate_message(message, escaped):
    """message field validator

    Messages can be single line or multi-line. In the multi-line case
    messages have the form:

    <heading>\n
    <object>

    Where heading is an optional string (escaped with normal escaping
    rules) and object is a structured representation of an object such
    as a table or protobuf. The only objects currently allowed are:

    * Tables (triton::common::table_printer)

    * Model config protobuf messages



    Parameters
    ----------
    message : str
        message portion of log record (may be multiple lines)
    escaped : bool
        whether the message is escaped

    Raises
    ------
    Exception If message is expected to be escaped but is not
    or object doesn't match formatting

    Examples
    --------

    validate_message("foo",escaped=True) -> Exception
    validate_message('"foo"', escaped=True) -> pass
    validate_message('"foo"\nfoo',escaped=True) -> Exception
    validate_message('"foo"\n+--------+---------+--------+\n' \
                     '| Model  | Version | Status |\n' \
                     '+--------+---------+--------+\n' \
                     '| simple | 1       | READY  |\n' \
                     '+--------+---------+--------+',
                      escaped=True) -> pass

    """

    split_message = message.split("\n")
    heading = split_message[0]
    obj = split_message[1:] if len(split_message) > 1 else []
    if heading and escaped:
        try:
            json.loads(heading)
        except Exception as e:
            raise Exception(
                f"{e.__class__.__name__} {e}\nFirst line of message in log record is not a valid JSON string"
            )
    elif heading:
        with pytest.raises(json.JSONDecodeError):
            json.loads(heading)
    if obj:
        match = table_border_regex.search(obj[0])
        if match:
            validate_table(obj)
        elif escaped:
            validate_protobuf(obj)
        else:
            # if not escaped and not table we can't
            # guarantee why type of object is present
            pass


class TestLogFormat:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        test_case_name = request.node.name
        self._server_options = {}
        self._server_options["log-verbose"] = INT32_MAX
        self._server_options["log-info"] = 1
        self._server_options["log-error"] = 1
        self._server_options["log-warning"] = 1
        self._server_options["log-format"] = "default"
        self._server_options["model-repository"] = test_model_directory
        self._server_process = None
        self._server_options["log-file"] = os.path.join(
            test_logs_directory, test_case_name + ".server.log"
        )

    def _shutdown_server(self):
        if self._server_process:
            self._server_process.kill()
            self._server_process.wait()

    def _launch_server(self, escaped=None):
        cmd = ["tritonserver"]

        for key, value in self._server_options.items():
            cmd.append(f"--{key}={value}")

        env = os.environ.copy()

        if escaped is not None and not escaped:
            env[ESCAPE_ENVIRONMENT_VARIABLE] = "0"
        elif escaped is not None and escaped:
            env[ESCAPE_ENVIRONMENT_VARIABLE] = "1"
        else:
            del env[ESCAPE_ENVIRONMENT_VARIABLE]
        log_file = self._server_options["log-file"]
        with open(f"{log_file}.stderr.log", "w") as output_err_:
            with open(f"{log_file}.stdout.log", "w") as output_:
                self._server_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output_,
                    stderr=output_err_,
                )

        wait_time = 5

        while wait_time and not os.path.exists(self._server_options["log-file"]):
            time.sleep(1)
            wait_time -= 1

        if not os.path.exists(self._server_options["log-file"]):
            raise Exception("Log not found")

        # Give server a little time to have the endpoints up and ready
        time.sleep(10)

    def _validate_log_record(self, record, format_regex, escaped):
        match = format_regex.search(record)
        assert match, "Invalid log line"

        for field, value in match.groupdict().items():
            if field not in validators:
                continue
            try:
                validators[field](value, escaped)
            except Exception as e:
                raise Exception(
                    f"{e.__class__.__name__} {e}\nInvalid {field}: '{match.group(field)}' in log record '{record}'"
                )

    def _parse_log_file(self, file_path, format_regex):
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
        log_records = [
            "".join(log_record_lines).rstrip("\n") for log_record_lines in log_records
        ]
        return log_records

    def _validate_log_file(self, file_path, format_regex, escaped):
        log_records = self._parse_log_file(file_path, format_regex)
        for log_record in log_records:
            self._validate_log_record(log_record, format_regex, escaped)

    def _detect_injection(self, log_records, injected_record):
        for record in log_records:
            if record == injected_record:
                raise LogInjectionError(
                    f"LOG INJECTION ATTACK! Found: {injected_record}"
                )

    @pytest.mark.parametrize(
        "log_format,format_regex",
        FORMATS,
        ids=IDS,
    )
    def test_format(self, log_format, format_regex):
        self._server_options["log-format"] = log_format.replace("_unescaped", "")

        escaped = "_unescaped" not in log_format

        self._launch_server(escaped)
        self._shutdown_server()
        self._validate_log_file(self._server_options["log-file"], format_regex, escaped)

    @pytest.mark.parametrize(
        "log_format,format_regex,injected_record",
        INJECTED_FORMATS,
        ids=INJECTED_IDS,
    )
    def test_injection(self, log_format, format_regex, injected_record):
        self._server_options["log-format"] = log_format.replace("_unescaped", "")

        escaped = "_unescaped" not in log_format

        self._launch_server(escaped)

        try:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=False
            )

            # TODO Refactor server launch, shutdown into reusable class
            wait_time = 10

            while wait_time:
                try:
                    if triton_client.is_server_ready():
                        break
                # Gracefully handle connection error if server endpoint isn't up yet
                except Exception as e:
                    print(
                        f"Client failed to connect, retries remaining: {wait_time}. Error: {e}"
                    )

                time.sleep(1)
                wait_time -= 1
                print(f"Server not ready yet, retries remaining: {wait_time}")

            while wait_time and not triton_client.is_model_ready("simple"):
                time.sleep(1)
                wait_time -= 1

            if not triton_client.is_server_ready():
                raise Exception("Server not Ready")

            if not triton_client.is_model_ready("simple"):
                raise Exception("Model not Ready")

        except Exception as e:
            self._shutdown_server()
            raise Exception(f"{e.__class__.__name__} {e}\ncontext creation failed")

        input_name = f"\n{injected_record}\n{injected_record}"

        input_data = numpy.random.randn(1, 3).astype(numpy.float32)
        input_tensor = httpclient.InferInput(input_name, input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)
        try:
            with pytest.raises(InferenceServerException):
                triton_client.infer(model_name="simple", inputs=[input_tensor])
        except Exception as e:
            raise Exception(f"{e.__class__.__name__} {e}\ninference failed")
        finally:
            self._shutdown_server()

        log_records = self._parse_log_file(
            self._server_options["log-file"], format_regex
        )

        if not escaped:
            with pytest.raises(LogInjectionError):
                self._detect_injection(log_records, injected_record)
        else:
            self._detect_injection(log_records, injected_record)
