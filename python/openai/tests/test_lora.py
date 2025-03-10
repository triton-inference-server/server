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
import shutil
import unittest

from huggingface_hub import snapshot_download
from openai import BadRequestError, NotFoundError

from .utils import OpenAIServer


def is_vllm_installed():
    try:
        import vllm as _

        return True
    except ImportError:
        return False


class LoRATest(unittest.TestCase):
    _model_name = "gemma-2b"
    # TODO: Find a LoRA model that has its own tokenizer.
    _tokenizer = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    _lora_separator = "_lora_"
    _prompt = "When was the wheel invented?"
    # more prompts that may yield different outputs:
    # - "Why can camels survive for long without water?"
    # - "What is LAPR?"
    # - "What is the difference between pets and cattle?"
    _temperature = 0
    _top_p = 1

    def setUp(self):
        self._completions_outputs = {}
        self._chat_completion_outputs = {}

    def _create_model_repository_with_lora(self):
        shutil.rmtree("models", ignore_errors=True)
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

    def _create_model_repository_without_lora(self):
        shutil.rmtree("models", ignore_errors=True)
        os.makedirs(f"models/{self._model_name}/1", exist_ok=True)
        with open(f"models/{self._model_name}/config.pbtxt", "w") as f:
            f.write('backend: "vllm"')
        with open(f"models/{self._model_name}/1/model.json", "w") as f:
            f.write(json.dumps({"model": "unsloth/gemma-2b"}))

    def _create_model_repository_mock_llm(self):
        shutil.rmtree("models", ignore_errors=True)
        os.makedirs(f"models/{self._model_name}/1", exist_ok=True)
        with open(f"models/{self._model_name}/config.pbtxt", "w") as f:
            f.write(
                """
                backend: "python"
                max_batch_size: 0
                model_transaction_policy { decoupled: True }
                input [
                    {
                        name: "text_input"
                        data_type: TYPE_STRING
                        dims: [ 1 ]
                    },
                    {
                        name: "stream"
                        data_type: TYPE_BOOL
                        dims: [ 1 ]
                    },
                    {
                        name: "sampling_parameters"
                        data_type: TYPE_STRING
                        dims: [ 1 ]
                    },
                    {
                        name: "exclude_input_in_output"
                        data_type: TYPE_BOOL
                        dims: [ 1 ]
                    }
                ]
                output [
                    {
                        name: "text_output"
                        data_type: TYPE_STRING
                        dims: [ -1 ]
                    }
                ]
            """
            )
        shutil.copy(
            "tests/test_models/mock_llm/1/model.py", f"models/{self._model_name}/1"
        )

    def _get_model_name(self, lora_name):
        model_name = self._model_name
        if lora_name != "":
            model_name += f"{self._lora_separator}{lora_name}"
        return model_name

    def _test_list_models(self, client, expected_lora_names):
        expected_model_names = []
        for lora_name in expected_lora_names:
            expected_model_names.append(self._get_model_name(lora_name))
        models = client.models.list()
        for model in models:
            self.assertIn(model.id, expected_model_names)
            expected_model_names.remove(model.id)
        self.assertEqual(len(expected_model_names), 0)

    def _test_retrieve_model(self, client, lora_name):
        model_name = self._get_model_name(lora_name)
        model = client.models.retrieve(model_name)
        self.assertEqual(model.id, model_name)

    def _test_completions(self, client, lora_name):
        model_name = self._get_model_name(lora_name)
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
        model_name = self._get_model_name(lora_name)
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
    def test_lora_separator_not_set(self):
        self._create_model_repository_with_lora()
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                self._tokenizer,
            ],
            env_dict={"CUDA_VISIBLE_DEVICES": "0"},
        ) as server:
            client = server.get_client()
            # Test listing/retrieving models
            self._test_list_models(client, [""])
            self._test_retrieve_model(client, "")
            with self.assertRaises(NotFoundError) as e:
                self._test_retrieve_model(client, "doll")
            expected_error = f"Error code: 404 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}doll'}}"
            self.assertEqual(str(e.exception), expected_error)
            with self.assertRaises(NotFoundError) as e:
                self._test_retrieve_model(client, "sheep")
            expected_error = f"Error code: 404 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}sheep'}}"
            self.assertEqual(str(e.exception), expected_error)
            # Test selecting LoRAs
            self._test_completions(client, "")
            self._test_chat_completion(client, "")
            with self.assertRaises(BadRequestError) as e:
                self._test_completions(client, "doll")
            expected_error = f"Error code: 400 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}doll'}}"
            self.assertEqual(str(e.exception), expected_error)
            with self.assertRaises(BadRequestError) as e:
                self._test_chat_completion(client, "sheep")
            expected_error = f"Error code: 400 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}sheep'}}"
            self.assertEqual(str(e.exception), expected_error)

    @unittest.skipUnless(is_vllm_installed(), "vLLM not installed")
    def test_lora_separator_set(self):
        self._create_model_repository_with_lora()
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                self._tokenizer,
                "--lora-separator",
                self._lora_separator,
            ],
            env_dict={"CUDA_VISIBLE_DEVICES": "0"},
        ) as server:
            client = server.get_client()
            # Test listing/retrieving models
            self._test_list_models(client, ["", "doll", "sheep"])
            self._test_retrieve_model(client, "")
            self._test_retrieve_model(client, "doll")
            self._test_retrieve_model(client, "sheep")
            # Test retrieving LoRAs unknown to the backend
            with self.assertRaises(NotFoundError) as e:
                self._test_retrieve_model(client, "unknown")
            expected_error = f"Error code: 404 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}unknown'}}"
            self.assertEqual(str(e.exception), expected_error)
            # Test selecting LoRAs
            self._test_completions(client, "")
            self._test_completions(client, "doll")
            self._test_completions(client, "sheep")
            self._test_chat_completion(client, "")
            self._test_chat_completion(client, "doll")
            self._test_chat_completion(client, "sheep")
            # Test selecting LoRAs unknown to the backend
            expected_error = f"Error code: 400 - {{'detail': 'Unknown LoRA: unknown; for model: {self._model_name}{self._lora_separator}unknown'}}"
            with self.assertRaises(BadRequestError) as e:
                self._test_completions(client, "unknown")
            self.assertEqual(str(e.exception), expected_error)
            with self.assertRaises(BadRequestError) as e:
                self._test_chat_completion(client, "unknown")
            self.assertEqual(str(e.exception), expected_error)

    @unittest.skipUnless(is_vllm_installed(), "vLLM not installed")
    def test_lora_separator_set_for_lora_off_model(self):
        self._create_model_repository_without_lora()
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                self._tokenizer,
                "--lora-separator",
                self._lora_separator,
            ],
            env_dict={"CUDA_VISIBLE_DEVICES": "0"},
        ) as server:
            client = server.get_client()
            # Test listing/retrieving models
            self._test_list_models(client, [""])
            self._test_retrieve_model(client, "")
            # Test retrieving models with LoRAs
            with self.assertRaises(NotFoundError) as e:
                self._test_retrieve_model(client, "doll")
            expected_error = f"Error code: 404 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}doll'}}"
            self.assertEqual(str(e.exception), expected_error)
            # Test inference
            self._test_completions(client, "")
            self._test_chat_completion(client, "")
            # Test selecting LoRAs
            expected_error = f"Error code: 400 - {{'detail': 'Unknown LoRA: sheep; for model: {self._model_name}{self._lora_separator}sheep'}}"
            with self.assertRaises(BadRequestError) as e:
                self._test_completions(client, "sheep")
            self.assertEqual(str(e.exception), expected_error)
            with self.assertRaises(BadRequestError) as e:
                self._test_chat_completion(client, "sheep")
            self.assertEqual(str(e.exception), expected_error)

    @unittest.skipUnless(is_vllm_installed(), "vLLM not installed")
    def test_lora_separator_set_for_non_vllm_formatted_models(self):
        self._create_model_repository_mock_llm()
        with OpenAIServer(
            cli_args=[
                "--model-repository",
                "models",
                "--tokenizer",
                self._tokenizer,
                "--backend",
                "vllm",
                "--lora-separator",
                self._lora_separator,
            ],
            env_dict={"CUDA_VISIBLE_DEVICES": "0"},
        ) as server:
            client = server.get_client()
            # Test listing/retrieving models
            self._test_list_models(client, [""])
            self._test_retrieve_model(client, "")
            # Test retrieving models with LoRAs
            with self.assertRaises(NotFoundError) as e:
                self._test_retrieve_model(client, "sheep")
            expected_error = f"Error code: 404 - {{'detail': 'Unknown model: {self._model_name}{self._lora_separator}sheep'}}"
            self.assertEqual(str(e.exception), expected_error)
            # Test selecting LoRAs
            # Expectation:
            #   If the frontend cannot determine which LoRA(s) are available, then any
            #   request with a well-formed LoRA model name will be inferenced.
            self._test_completions(client, "doll")
            self._test_chat_completion(client, "doll")


if __name__ == "__main__":
    unittest.main()
