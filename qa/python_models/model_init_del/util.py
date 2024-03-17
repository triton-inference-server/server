#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import fcntl
import os

_model_name = "model_init_del"

#
# Helper functions for reading/writing state to disk
#


def _get_number(filename):
    full_path = os.path.join(os.environ["MODEL_LOG_DIR"], filename)
    try:
        with open(full_path, mode="r", encoding="utf-8", errors="strict") as f:
            fcntl.lockf(f, fcntl.LOCK_SH)
            txt = f.read()
    except FileNotFoundError:
        txt = "0"
    return int(txt)


def _store_number(filename, number):
    full_path = os.path.join(os.environ["MODEL_LOG_DIR"], filename)
    txt = str(number)
    with open(full_path, mode="w", encoding="utf-8", errors="strict") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write(txt)


def _inc_number(filename):
    full_path = os.path.join(os.environ["MODEL_LOG_DIR"], filename)
    try:
        with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            txt = f.read()
            number = int(txt) + 1
            txt = str(number)
            f.truncate(0)
            f.seek(0)
            f.write(txt)
    except FileNotFoundError:
        number = 1
        _store_number(filename, number)
    return number


#
# Functions for communicating initialize and finalize count between the model
# and test
#


def _get_count_filename(kind):
    if kind != "initialize" and kind != "finalize":
        raise KeyError("Invalid count kind: " + str(kind))
    filename = _model_name + "_" + kind + "_count.txt"
    return filename


def get_count(kind):
    return _get_number(_get_count_filename(kind))


def inc_count(kind):
    return _inc_number(_get_count_filename(kind))


def reset_count(kind):
    count = 0
    _store_number(_get_count_filename(kind), count)
    return count


#
# Functions for communicating varies of delay (in seconds) to the model
#


def _get_delay_filename(kind):
    if kind != "initialize" and kind != "infer":
        raise KeyError("Invalid delay kind: " + str(kind))
    filename = _model_name + "_" + kind + "_delay.txt"
    return filename


def get_delay(kind):
    return _get_number(_get_delay_filename(kind))


def set_delay(kind, delay):
    _store_number(_get_delay_filename(kind), delay)
    return delay


#
# Functions for modifying the model
#


def update_instance_group(instance_group_str):
    full_path = os.path.join(os.path.dirname(__file__), "config.pbtxt")
    with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
        txt = f.read()
        txt, post_match = txt.split("instance_group [")
        txt += "instance_group [\n"
        txt += instance_group_str
        txt += "\n]  # end instance_group\n"
        txt += post_match.split("\n]  # end instance_group\n")[1]
        f.truncate(0)
        f.seek(0)
        f.write(txt)
    return txt


def update_sequence_batching(sequence_batching_str):
    full_path = os.path.join(os.path.dirname(__file__), "config.pbtxt")
    with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
        txt = f.read()
        if "sequence_batching {" in txt:
            txt, post_match = txt.split("sequence_batching {")
            if sequence_batching_str != "":
                txt += "sequence_batching {\n"
                txt += sequence_batching_str
                txt += "\n}  # end sequence_batching\n"
            txt += post_match.split("\n}  # end sequence_batching\n")[1]
        elif sequence_batching_str != "":
            txt += "\nsequence_batching {\n"
            txt += sequence_batching_str
            txt += "\n}  # end sequence_batching\n"
        f.truncate(0)
        f.seek(0)
        f.write(txt)
    return txt


def update_model_file():
    full_path = os.path.join(os.path.dirname(__file__), "1", "model.py")
    with open(full_path, mode="a", encoding="utf-8", errors="strict") as f:
        f.write("\n# dummy model file update\n")


def enable_batching():
    full_path = os.path.join(os.path.dirname(__file__), "config.pbtxt")
    with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
        txt = f.read()
        txt = txt.replace("max_batch_size: 0", "max_batch_size: 2")
        f.truncate(0)
        f.seek(0)
        f.write(txt)
    return txt


def disable_batching():
    full_path = os.path.join(os.path.dirname(__file__), "config.pbtxt")
    with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
        txt = f.read()
        txt = txt.replace("max_batch_size: 2", "max_batch_size: 0")
        f.truncate(0)
        f.seek(0)
        f.write(txt)
    return txt
