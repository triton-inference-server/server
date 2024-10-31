# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from enum import Enum
from typing import List, Union

from _error_mapping import handle_triton_error
from pydantic import FieldValidationInfo, field_validator
from pydantic.dataclasses import dataclass


# 1-to-1 copy of RestrictedCategory Enum from https://github.com/triton-inference-server/server/blob/main/src/restricted_features.h
class Protocols(Enum):
    HEALTH = "health"
    METADATA = "metadata"
    INFERENCE = "inference"
    SHM_MEMORY = "shared-memory"
    MODEL_CONFIG = "model-config"
    MODEL_REPOSITORY = "model-repository"
    STATISTICS = "statistics"
    TRACE = "trace"
    LOGGING = "logging"


@dataclass
class RestrictedFeatureGroup:
    key: str
    value: str
    protocols: List[Protocols]


class RestrictedFeatures:
    Features = Protocols
    Group = RestrictedFeatureGroup

    def __init__(self):
        self.FeatureGroups
