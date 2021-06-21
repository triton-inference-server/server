<!--
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Benchmarking with NVIDIA Triton Inference Server GKE Marketplace Application

**Table Of Contents**
- [Models](#models)
- [Performance](#performance)

## Models

First, we collect a set of TensorFlow and TensorRT models to compare:

- Get [Distill Bert fine-tuned with Squad Q&A task](https://huggingface.co/distilbert-base-cased-distilled-squad/tree/main) from Huggingface. `wget https://huggingface.co/distilbert-base-cased-distilled-squad/blob/main/saved_model.tar.gz`
- Get [Bert base fine-tuned with Squad Q&A task](https://huggingface.co/deepset/bert-base-cased-squad2/tree/main) from Huggingface `wget https://huggingface.co/deepset/bert-base-cased-squad2/blob/main/saved_model.tar.gz`
- Follow [TensorRT Demo Bert](https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT) to convert BERT base model to TensorRT Engine, choose sequence length of 384 to match previous 2 TensorFlow models. Last step, we choose to create TensorRT engine with 2 optimization profile, profile 0 for batch size 1 and profile 1 for batch size 4 run: `python3 builder.py -m models/fine-tuned/bert_tf_ckpt_base_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/model.plan -b 8 -s 384 --fp16 --int8 --strict -c models/fine-tuned/bert_tf_ckpt_base_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v2.0.json -v models/fine-tuned/bert_tf_ckpt_base_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln -imh`. This needs to be ran on the inference GPU respectively (Engine optimized with A100 cannot be used for inference on T4).

We the place the model into a GCS with following structure, `config.pbtxt` was provided.
```
    ├── bert_base_trt_gpu
    │   ├── 1
    │   │   └── model.plan
    │   └── config.pbtxt
    ├── bert_base_trt_gpu_seqlen128
    │   ├── 1
    │   │   └── model.plan
    │   └── config.pbtxt    
    ├── bert_base_tf_gpu
    │   ├── 1
    │   │   └── model.savedmodel
    │   └── config.pbtxt      
    ├── bert_base_tf_cpu
    │   ├── 1
    │   │   └── model.savedmodel
    │   └── config.pbtxt
    ├── bert_distill_tf_gpu 
    │   ├── 1
    │   │   └── model.savedmodel
    │   └── config.pbtxt
    └── bert_distill_tf_cpu
        ├── 1
        │   └── model.savedmodel
        └── config.pbtxt 
```

When deploy Triton GKE application, point the model repository to directory contains the structure above with actual models. 

## Performance

We use perf analyzer of Triton to benchmark the performance of each model, the perf analyzer reside in another pod of the GKE cluster. 
```bash
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
bash perf_query.sh 35.194.5.119:80 bert_base_trt_gpu 384
```

We deploy model on n1-standard-96 for CPU BERT BASE and Distill BERT and (n1-standard-4 + T4) for GPU BERT models, the sequence length  of the BERT model is 384 token, and measure the latency/throughput with a concurrency sweep with Triton's performance analyzer. The latency includes Istio ingress/load balancing and reflect the true round trip cost in the same GCP zone.

For all the model with sequence length of 384:
CPU BERT BASE: latency: 700ms, throughput: 12 qps
CPU Distill BERT: latency: 369ms, throughput: 24 qps

GPU BERT BASE: latency: 230ms, throughput: 34.7 qps
GPU Distill BERT: latency: 118ms, throughput: 73.3 qps
GPU TensorRT BERT BASE: latency: 50ms, throughput: 465 qps

With n1-standard-96 priced at $4.56/hr and n1-standard-4 at $0.19/hr and T4 at $0.35/hr totaling $0.54/hr. While achieving a much lower latency, the TCO of BERT inference with TensorRT on T4 is over 163 times that of Distill BERT inference on n1-standard-96.

  

 