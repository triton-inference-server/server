# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from argparse import ArgumentParser

from tensorrt_llm import BuildConfig
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.plugin import PluginConfig


def generate_model_engine(model: str, engines_path: str):
    config = BuildConfig(plugin_config=PluginConfig.from_dict({"_gemm_plugin": "auto"}))

    lora_config = LoraConfig(
        lora_target_modules=["attn_q", "attn_k", "attn_v"],
        max_lora_rank=8,
        max_loras=4,
        max_cpu_loras=8,
    )

    engine = LLM(
        model,
        dtype="float16",
        max_batch_size=128,
        build_config=config,
        guided_decoding_backend="xgrammar",
        lora_config=lora_config,
    )

    engine.save(engines_path)
    engine.shutdown()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", "-m", help="model huggingface id or path to the model"
    )
    parser.add_argument("--engine_path", "-e", help="directory of the output engine")
    FLAGS = parser.parse_args()

    generate_model_engine(FLAGS.model, FLAGS.engine_path)
    print(f"model {FLAGS.model}'s engine has been saved to {FLAGS.engine_path}")
