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


from __future__ import annotations

import os
import time
from typing import List

import tritonserver
from engine.engine import OpenAIEngine
from schemas.openai import Model, ObjectType
from utils.tokenizer import get_tokenizer
from utils.triton import TritonModelMetadata, determine_request_converter


class TritonOpenAIEngine(OpenAIEngine):
    def __init__(self, server: tritonserver.Server):
        # Assume an already configured and started server
        self.server = server

        # NOTE: Creation time and model metadata will be static at startup for
        # now, and won't account for dynamically loading/unloading models.
        self.create_time = int(time.time())
        self.model_metadata = self._get_model_metadata()

    def live(self) -> bool:
        return self.server.live()

    def metrics(self) -> str:
        return self.server.metrics()

    def models(self) -> List[Model]:
        models = []
        for metadata in self.model_metadata:
            models.append(
                Model(
                    id=metadata.name,
                    created=metadata.create_time,
                    object=ObjectType.model,
                    owned_by="Triton Inference Server",
                ),
            )

        return models

    def _get_tokenizer(self):
        # TODO: Consider support for custom tokenizers
        tokenizer = None
        tokenizer_name = os.environ.get("TOKENIZER")
        if tokenizer_name:
            print(
                f"Using env var TOKENIZER={tokenizer_name} to determine the tokenizer"
            )
            tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def _get_model_metadata(self) -> List[TritonModelMetadata]:
        tokenizer = self._get_tokenizer()

        # One tokenizer, convert function, and creation time for all loaded models.
        # NOTE: This doesn't currently support having both a vLLM and TRT-LLM
        # model loaded at the same time.
        model_metadata = []

        # Read all triton models and gather the respective backends of each
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            backend = model.config()["backend"]
            print(f"Found model: {name=}, {backend=}")

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=tokenizer,
                create_time=self.create_time,
                request_converter=determine_request_converter(backend),
            )
            model_metadata.append(metadata)

        return model_metadata
