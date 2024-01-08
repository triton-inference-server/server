#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest

import torch

# satisfy Python runtime import requirements
sys.modules["triton_python_backend_utils"] = unittest.mock.MagicMock()
# import modules from Python runtime to be tested
from py_runtime import _gather_torch_tensors, _scatter_torch_tensors


class PyTorchPythonBackendRuntimeUnittest(unittest.TestCase):
    # _gather_scatter_cases: [(tensors_scatter, tensors_gather, sections), ...]
    #   tensors_scatter: [an_infer_request, ...]
    #     an_infer_request: [a_torch_tensor_with_batch_dimension, ...]
    #   tensors_gather: [a_torch_tensor_gathering_all_requests, ...]
    #   sections: [batch_size_of_the_corresponding_infer_request, ...]
    _gather_scatter_cases = [
        # shape [batch=1, 1]
        ([[torch.tensor([[1]])]], [torch.tensor([[1]])], [1]),
        # shape [batch=1, 2]
        ([[torch.tensor([[1, 2]])]], [torch.tensor([[1, 2]])], [1]),
        # shape [batch=1, 2, 4]
        ([[torch.arange(8).reshape(1, 2, 4)]], [torch.arange(8).reshape(1, 2, 4)], [1]),
        # shape [batch=3, 1]
        ([[torch.arange(3).reshape(3, 1)]], [torch.arange(3).reshape(3, 1)], [3]),
        # shapes ([batch=1, 1], [batch=1, 2])
        (
            [[torch.tensor([[1]]), torch.tensor([[2, 3]])]],
            [torch.tensor([[1]]), torch.tensor([[2, 3]])],
            [1],
        ),
        # scatter shape [batch=1, 1] x 2 -> gather shape [batch=2, 1]
        (
            [[torch.tensor([[1]])], [torch.tensor([[2]])]],
            [torch.tensor([[1], [2]])],
            [1, 1],
        ),
        # scatter shape [batch=1, 2, 1] x 3 -> gather shape [batch=3, 2, 1]
        (
            [[torch.tensor([[[i], [i + 3]]])] for i in range(3)],
            [torch.tensor([[[0], [3]], [[1], [4]], [[2], [5]]])],
            [1, 1, 1],
        ),
        # scatter shape [batch=1, 1] & [batch=2, 1] -> gather shape [batch=3, 1]
        (
            [[torch.tensor([[1]])], [torch.tensor([[2], [3]])]],
            [torch.tensor([[1], [2], [3]])],
            [1, 2],
        ),
        # scatter shape [batch=3, 1, 1] & [batch=1, 1, 1] & [batch=2, 1, 1]
        # -> gather shape [batch=6, 1, 1]
        (
            [
                [torch.tensor([[[0]], [[1]], [[2]]])],
                [torch.tensor([[[3]]])],
                [torch.tensor([[[4]], [[5]]])],
            ],
            [torch.arange(6).reshape(6, 1, 1)],
            [3, 1, 2],
        ),
        # scatter shapes ([batch=3, 1, 1], [batch=3, 2]) & ([batch=2, 1, 1], [batch=2, 2])
        # -> gather shapes ([batch=5, 1, 1], [batch=5, 2])
        (
            [
                [
                    torch.tensor([[[0]], [[1]], [[2]]]),
                    torch.tensor([[5, 6], [7, 8], [9, 10]]),
                ],
                [torch.tensor([[[3]], [[4]]]), torch.tensor([[11, 12], [13, 14]])],
            ],
            [
                torch.arange(5).reshape(5, 1, 1),
                torch.arange(start=5, end=15).reshape(5, 2),
            ],
            [3, 2],
        ),
    ]

    def test_gather_torch_tensors(self):
        for (
            tensors_scatter,
            expected_tensors_gather,
            expected_sections,
        ) in self._gather_scatter_cases:
            tensors_gather, sections = _gather_torch_tensors(tensors_scatter)

            self.assertIsInstance(tensors_gather, list)
            self.assertEqual(len(tensors_gather), len(expected_tensors_gather))
            for j in range(len(expected_tensors_gather)):
                expected_tensor = expected_tensors_gather[j]
                tensor = tensors_gather[j]
                self.assertIsInstance(tensor, torch.Tensor)
                self.assertTrue(torch.equal(tensor, expected_tensor))

            self.assertIsInstance(sections, list)
            self.assertEqual(len(sections), len(expected_sections))
            for i in range(len(expected_sections)):
                expected_section = expected_sections[i]
                section = sections[i]
                self.assertIsInstance(section, int)
                self.assertEqual(section, expected_section)

    def test_scatter_torch_tensors(self):
        for (
            expected_tensors_scatter,
            tensors_gather,
            sections,
        ) in self._gather_scatter_cases:
            tensors_scatter = _scatter_torch_tensors(tensors_gather, sections)
            self.assertIsInstance(tensors_scatter, list)
            self.assertEqual(len(tensors_scatter), len(expected_tensors_scatter))
            for i in range(len(expected_tensors_scatter)):
                expected_tensors = expected_tensors_scatter[i]
                tensors = tensors_scatter[i]
                self.assertIsInstance(tensors, list)
                self.assertEqual(len(tensors), len(expected_tensors))
                for j in range(len(expected_tensors)):
                    expected_tensor = expected_tensors[j]
                    tensor = tensors[j]
                    self.assertIsInstance(tensor, torch.Tensor)
                    self.assertTrue(torch.equal(tensor, expected_tensor))


if __name__ == "__main__":
    unittest.main()
