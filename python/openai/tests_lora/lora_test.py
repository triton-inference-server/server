# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import unittest

from huggingface_hub import snapshot_download
from openai import BadRequestError

from .server_wrapper import OpenAIServer


def is_vllm_installed():
    try:
        import vllm as _

        return True
    except ImportError:
        return False


class LoRATest(unittest.TestCase):
    _model_name = "gemma-2b"
    _lora_separator = "_lora_"
    _prompt = "When was the wheel invented?"
    # more prompts that may yield different outputs:
    # - "Why can camels survive for long without water?"
    # - "What is LAPR?"
    # - "What is the difference between pets and cattle?"
    _temperature = 0
    _top_p = 1

    def setUp(self):
        self._create_model_repository()
        self._completions_outputs = {}
        self._chat_completion_outputs = {}

    def _create_model_repository(self):
        os.makedirs(f"models/{self._model_name}/1", exist_ok=True)
        with open(f"models/{self._model_name}/config.pbtxt", "w") as f:
            f.write('backend: "vllm"')
        with open(f"models/{self._model_name}/1/model.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "model": "unsloth/gemma-2b",
                        "enable_lora": True,
                        "max_lora_rank": 32,
                    }
                )
            )
        with open(f"models/{self._model_name}/1/multi_lora.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "doll": f"models/{self._model_name}/1/GemmaDoll",
                        "sheep": f"models/{self._model_name}/1/GemmaSheep",
                    }
                )
            )
        snapshot_download(
            repo_id="swathijn/GemmaDoll-2b-dolly-LORA-Tune",
            local_dir=f"models/{self._model_name}/1/GemmaDoll",
        )
        snapshot_download(
            repo_id="eduardo-alvarez/GemmaSheep-2B-LORA-TUNED",
            local_dir=f"models/{self._model_name}/1/GemmaSheep",
        )

    def _test_completions(self, client, lora_name):
        model_name = self._model_name
        if lora_name != "":
            model_name += f"{self._lora_separator}{lora_name}"
        completion = client.completions.create(
            model=model_name,
            prompt=self._prompt,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        self.assertEqual(completion.model, model_name)
        output = completion.choices[0].text
        for other_output in self._completions_outputs.values():
            self.assertNotEqual(
                output,
                other_output,
                msg=f"other completions outputs: {self._completions_outputs}",
            )
        self._completions_outputs[lora_name] = output

    def _test_chat_completion(self, client, lora_name):
        model_name = self._model_name
        if lora_name != "":
            model_name += f"{self._lora_separator}{lora_name}"
        messages = [{"role": "user", "content": self._prompt}]
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        self.assertEqual(chat_completion.model, model_name)
        output = chat_completion.choices[0].message.content
        for other_output in self._chat_completion_outputs.values():
            self.assertNotEqual(
                output,
                other_output,
                msg=f"other chat outputs: {self._chat_completion_outputs}",
            )
        self._chat_completion_outputs[lora_name] = output

    @unittest.skipUnless(is_vllm_installed(), "vLLM not installed")
    def test_lora_separator_vllm_not_set(self):
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ]
        ) as server:
            client = server.get_client()
            # Check not selecting LoRA works
            self._test_completions(client, "")
            self._test_chat_completion(client, "")
            # Check selecting LoRA results in model not found
            with self.assertRaises(BadRequestError) as e:
                self._test_completions(client, "doll")
            self.assertEqual(
                str(e.exception),
                f"Error code: 400 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}doll'}}",
            )
            with self.assertRaises(BadRequestError) as e:
                self._test_chat_completion(client, "sheep")
            self.assertEqual(
                str(e.exception),
                f"Error code: 400 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}sheep'}}",
            )

    @unittest.skipUnless(is_vllm_installed(), "vLLM not installed")
    def test_lora_separator_vllm_set(self):
        # TODO: Find a model with LoRAs that has a tokenizer.
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "--model-and-lora-name-separator",
                self._lora_separator,
            ]
        ) as server:
            client = server.get_client()
            # Check selecting LoRA works
            self._test_completions(client, "")
            self._test_completions(client, "doll")
            self._test_completions(client, "sheep")
            self._test_chat_completion(client, "")
            self._test_chat_completion(client, "doll")
            self._test_chat_completion(client, "sheep")
            # Check selecting unknown LoRA results in LoRA not found
            # TODO: Server hangs when shutting down if LoRA not found.
            # expected_error_start = (
            #    "Error code: 400 - {'detail': '(\"LoRA unknown is not supported"
            # )
            # with self.assertRaises(BadRequestError) as e:
            #    self._test_completions(client, "unknown")
            # self.assertTrue(str(e.exception).startswith(expected_error_start))
            # with self.assertRaises(BadRequestError) as e:
            #    self._test_chat_completion(client, "unknown")
            # self.assertTrue(str(e.exception).startswith(expected_error_start))


if __name__ == "__main__":
    unittest.main()
