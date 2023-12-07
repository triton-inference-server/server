<!--
# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Instruction to create BERT engine for each Triton update

## Description

```
docker run --gpus all -it --network host \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~:/scripts nvcr.io/nvidia/tensorrt:23.12-py3

pip install onnx six torch tf2onnx tensorflow

git clone -b main https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive

export TRT_OSSPATH=/workspace/TensorRT
export TRT_LIBPATH=/lib/x86_64-linux-gnu

pushd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && ln -s ngc-cli/ngc ngc && echo "no-apikey\nascii\n" | ngc config set

popd

cd /workspace/TensorRT/demo/BERT
bash ./scripts/download_squad.sh
bash ./scripts/download_model.sh large 128
# bash ./scripts/download_model.sh large 384

mkdir -p engines

python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/model.ckpt -o engines/bert_large_int8_bs1_s128.engine -b 1 -s 128 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/ -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt --int8 --fp16 --strict --calib-num 1 -iln -imh

gsutil cp bert_large_int8_bs1_s128.engine gs://triton_sample_models/23_12/bert/1/model.plan
```

For each Triton upgrade, container version used to generate the model, and the model path in GCS `gs://triton_sample_models/23_12/` should be updated accordingly with the correct version.
