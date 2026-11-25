# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate model repositories for the response-cache key tests."""

import argparse
import shutil
from pathlib import Path

CONFIG = """name: "cache_key"
backend: "python"
max_batch_size: 0
input [{ name: "INPUT" data_type: TYPE_INT32 dims: [-1, -1] }]
output [
  { name: "SUM" data_type: TYPE_INT64 dims: [-1] },
  { name: "MAX" data_type: TYPE_INT32 dims: [-1] }
]
instance_group [{ kind: KIND_CPU }]
response_cache { enable: true }
"""


def create_model(repository, name, batched=False, enabled=True, maxima=False):
    target = repository / name
    (target / "1").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).with_name("cache_key_model.py"), target / "1/model.py"
    )
    text = CONFIG.replace('name: "cache_key"', f'name: "{name}"')
    if batched:
        text = text.replace("max_batch_size: 0", "max_batch_size: 8")
        text = text.replace("dims: [-1]", "dims: [1]")
        text = text.replace("dims: [-1, -1]", "dims: [-1]")
    if not enabled:
        text = text.replace("enable: true", "enable: false")
    if maxima:
        text += 'parameters { key: "reduction" value { string_value: "max" } }\n'
    (target / "config.pbtxt").write_text(text)


def create_ensemble(repository, name, component, enabled=True):
    target = repository / name
    (target / "1").mkdir(parents=True)
    text = CONFIG.replace('name: "cache_key"', f'name: "{name}"')
    text = text.replace('backend: "python"', 'platform: "ensemble"')
    text = text.replace("instance_group [{ kind: KIND_CPU }]", "")
    if not enabled:
        text = text.replace("enable: true", "enable: false")
    text += 'ensemble_scheduling { step [{ model_name: "' + component + '" '
    text += 'model_version: -1 input_map { key: "INPUT" value: "INPUT" } '
    for output in ("SUM", "MAX"):
        text += f'output_map {{ key: "{output}" value: "{output}" }} '
    text += "}] }\n"
    (target / "config.pbtxt").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for repoA and repoB")
    root = parser.parse_args().output_dir
    create_model(root / "repoA", "cache_key")
    create_model(root / "repoA", "cache_key_batched", batched=True)
    create_model(root / "repoA", "cache_key_uncached", enabled=False)
    create_ensemble(root / "repoA", "cache_key_ensemble", "cache_key_uncached")
    create_model(root / "repoA", "X")
    create_model(root / "repoB", "X", maxima=True)
    create_ensemble(root / "repoA", "ensA", "X", enabled=False)
    create_ensemble(root / "repoB", "ensB", "X", enabled=False)


if __name__ == "__main__":
    main()
