# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

import gradio as gr
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--triton_url", default="localhost:8001")
args = parser.parse_args()

client = grpcclient.InferenceServerClient(url=f"{args.triton_url}")


def generate(prompt):
    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))
    input_text = grpcclient.InferInput("prompt", text_obj.shape,
                                       np_to_triton_dtype(text_obj.dtype))
    input_text.set_data_from_numpy(text_obj)

    output_img = grpcclient.InferRequestedOutput("generated_image")

    response = client.infer(model_name="pipeline",
                            inputs=[input_text],
                            outputs=[output_img])
    resp_img = response.as_numpy("generated_image")
    print(resp_img.shape)
    im = Image.fromarray(np.squeeze(resp_img.astype(np.uint8)))
    return im


with gr.Blocks() as app:
    prompt = gr.Textbox(label="Prompt")
    submit_btn = gr.Button("Generate")
    img_output = gr.Image().style(height=512)
    submit_btn.click(fn=generate, inputs=prompt, outputs=img_output)

app.launch(server_name="0.0.0.0")
