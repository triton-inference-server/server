..
  # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

.. _section-capabilities:

Capabilities
============

The following table shows which backends support each major inference
server feature. See :ref:`section-datatypes` for information on
data-types supported by each backend.

+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Feature                  |TensorRT |TensorFlow |Caffe2 |ONNX Runtime |PyTorch |Custom |
+=========================+=========+===========+=======+=============+========+=======+
|Multi-GPU                |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Multi-Model              |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Batching                 |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Dynamic Batching         |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Sequence Batching        |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Variable-Size Tensors    |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Tensor Reshape           |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|String Datatype          |         |Yes        |       |Yes          |        |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|HTTP API                 |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|GRPC API                 |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|GRPC Streaming API       |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Ensembling               |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|Shared Memory API        |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
|CUDA Shared Memory API   |Yes      |Yes        |Yes    |Yes          |Yes     |Yes    |
+-------------------------+---------+-----------+-------+-------------+--------+-------+
