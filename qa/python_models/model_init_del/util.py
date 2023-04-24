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

import os

__model_name = "model_init_del"

#
# Helper functions for reading/writing state to disk
#

def __get_count(filename):
    full_path = os.path.join(os.environ["MODEL_LOG_DIR"], filename)
    try:
        with open(full_path, mode="r", encoding="utf-8", errors="strict") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = "0"
    return int(txt)

def __store_count(filename, number):
    full_path = os.path.join(os.environ["MODEL_LOG_DIR"], filename)
    txt = str(number)
    with open(full_path, mode="w", encoding="utf-8", errors="strict") as f:
        f.write(txt)

def __inc_count(filename):
    count = __get_count(filename)
    count += 1
    __store_count(filename, count)
    return count

def __reset_count(filename):
    count = 0
    __store_count(filename, count)
    return count

#
# Functions for communicating initialize and finalize count between the model
# and test
#

__initialize_count_filename = __model_name + "_initialize_count.txt"
__finalize_count_filename = __model_name + "_finalize_count.txt"

def get_initialize_count():
    return __get_count(__initialize_count_filename)

def inc_initialize_count():
    return __inc_count(__initialize_count_filename)

def reset_initialize_count():
    return __reset_count(__initialize_count_filename)

def get_finalize_count():
    return __get_count(__finalize_count_filename)

def inc_finalize_count():
    return __inc_count(__finalize_count_filename)

def reset_finalize_count():
    return __reset_count(__finalize_count_filename)

#
# Functions for communicating varies of delay (in seconds) to the model
#

__initialize_delay_filename = __model_name + "_initialize_delay.txt"
__finalize_delay_filename = __model_name + "_finalize_delay.txt"

def get_initialize_delay():
    delay = __get_count(__initialize_delay_filename)
    return delay

def set_initialize_delay(delay):
    __store_count(__initialize_delay_filename, delay)
    return delay

def get_infer_delay():
    delay = __get_count(__finalize_delay_filename)
    return delay

def set_infer_delay(delay):
    __store_count(__finalize_delay_filename, delay)
    return delay

#
# Functions for modifying the model
#

def update_instance_group(instance_group_str):
    full_path = os.path.join(os.path.dirname(__file__), "config.pbtxt")
    with open(full_path, mode="r+", encoding="utf-8", errors="strict") as f:
        txt = f.read()
        txt = txt.split("instance_group [")[0]
        txt += "instance_group [\n"
        txt += instance_group_str
        txt += "\n]\n"
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
