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

import json
from copy import deepcopy
from enum import Enum
from typing import List, Union

import tritonserver
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from tritonfrontend._api._error_mapping import handle_triton_error
from tritonfrontend._c.tritonfrontend_bindings import InvalidArgumentError


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
class FeatureGroup:
    key: str
    value: str
    protocols: List[Protocols]

    @field_validator("protocols", mode="before")
    def validate_protocols(protocols: List[Protocols]) -> List[Protocols]:
        invalid_protocols = [
            item for item in protocols if not isinstance(item, Protocols)
        ]
        if invalid_protocols:
            raise tritonserver.InvalidArgumentError(
                f"Invalid protocols found: {invalid_protocols}. "
                f"Each item in 'protocols' should be an instance of the tritonfrontend.Protocols. "
                f"Valid options are: {[str(p) for p in Protocols]}"
            )
        return protocols


class RestrictedFeatures:
    def __init__(self, groups: List[FeatureGroup] = []):
        self.feature_groups = []
        self.protocols_restricted = set()

        for feat_group in groups:
            self.add_feature_group(feat_group)

    @handle_triton_error
    def add_feature_group(self, group: FeatureGroup):
        """
        Need to check for collision with protocols_restricted.
        If collision, raise InvalidArgumentError().
        If no collision, add group to feature_groups
        """
        for protocol in group.protocols:
            if protocol in self.protocols_restricted:
                raise InvalidArgumentError(
                    "A given protocol can only belong to one group."
                    f"{str(protocol)} already belongs to an existing group."
                )

        self.protocols_restricted.update(group.protocols)
        self.feature_groups.append(group)

    @handle_triton_error
    def create_feature_group(self, key: str, value: str, protocols: List[Protocols]):
        group = FeatureGroup(key, value, protocols)
        self.add_feature_group(group)

    @handle_triton_error
    def _gather_rest_data(self) -> dict:
        # Dataclass_Instance.__dict__ provides shallow copy, so need a deep copy IF modifying
        rfeat_data = [
            deepcopy(feat_group.__dict__) for feat_group in self.feature_groups
        ]

        for idx in range(len(rfeat_data)):
            rfeat_data[idx]["protocols"] = [
                feat.value for feat in rfeat_data[idx]["protocols"]
            ]

        return rfeat_data

    def __str__(self) -> str:
        """
        A function to retrieve user-friendly string to view object contents.
        """
        return json.dumps(self._gather_rest_data(), indent=2)

    def __repr__(self) -> str:
        """
        A function to retrieve representation that has not been formatted.
        """
        return json.dumps(self._gather_rest_data())
