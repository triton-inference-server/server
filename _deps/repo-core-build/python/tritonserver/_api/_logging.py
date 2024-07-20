# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Logging Utilities"""
import inspect
import os

from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import TRITONSERVER_LogMessage


def LogMessage(level: LogLevel, message: str):
    """Log Message using Triton Inference Server Logger

    Parameters
    ----------
    level : LogLevel
        log level one of LogLevel.WARN, LogLevel.ERROR, LogLevel.INFO
    message : str
        message

    Examples
    --------

    LogMessage(LogLevel.ERROR,"I've got a bad feeling about this ...")

    """

    filename, line_number = "unknown", -1
    try:
        current_frame = inspect.stack()[-1]
        filename, line_number = (
            os.path.basename(current_frame.filename),
            current_frame.lineno,
        )
    except Exception as e:
        pass
    TRITONSERVER_LogMessage(level, filename, line_number, message)
