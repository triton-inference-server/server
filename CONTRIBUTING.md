<!--
# Copyright 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Contribution Guidelines

Contributions that fix documentation errors or that make small changes
to existing code can be contributed directly by following the rules
below and submitting an appropriate PR.

Contributions intended to add significant new functionality must
follow a more collaborative path described in the following
points. Before submitting a large PR that adds a major enhancement or
extension, be sure to submit a GitHub issue that describes the
proposed change so that the Triton team can provide feedback.

- As part of the GitHub issue discussion, a design for your change
  will be agreed upon. An up-front design discussion is required to
  ensure that your enhancement is done in a manner that is consistent
  with Triton's overall architecture.

- The Triton project is spread across multiple repos. The Triton team
  will provide guidance about how and where your enhancement should be
  implemented.

- [Testing](docs/customization_guide/test.md) is a critical part of any Triton
  enhancement. You should plan on spending significant time on
  creating tests for your change. The Triton team will help you to
  design your testing so that it is compatible with existing testing
  infrastructure.

- If your enhancement provides a user visible feature then you need to
  provide documentation.

# Contribution Rules

- The code style convention is enforced by clang-format. See below on
  how to ensure your contributions conform. In general please follow
  the existing conventions in the relevant file, submodule, module,
  and project when you add new code or when you extend/fix existing
  functionality.

- Avoid introducing unnecessary complexity into existing code so that
  maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:

  - Avoid committing commented-out code.

  - Wherever possible, each PR should address a single concern. If
    there are several otherwise-unrelated things that should be fixed
    to reach a desired endpoint, it is perfectly fine to open several
    PRs and state in the description which PR depends on another
    PR. The more complex the changes are in a single PR, the more time
    it will take to review those changes.

  - Make sure that the build log is clean, meaning no warnings or
    errors should be present.

- Make sure all `L0_*` tests pass:

  - In the `qa/` directory, there are basic sanity tests scripted in
    directories named `L0_...`.  See the [Test](docs/customization_guide/test.md)
    documentation for instructions on running these tests.

- Triton Inference Server's default build assumes recent versions of
  dependencies (CUDA, TensorFlow, PyTorch, TensorRT,
  etc.). Contributions that add compatibility with older versions of
  those dependencies will be considered, but NVIDIA cannot guarantee
  that all possible build configurations work, are not broken by
  future contributions, and retain highest performance.

- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code). You need
  to complete the CLA described below before your PR can be merged.

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

# Coding Convention

All pull requests are checked against the
[pre-commit hooks](https://github.com/pre-commit/pre-commit-hooks)
located [in the repository's top-level .pre-commit-config.yaml](.pre-commit-config.yaml).
The hooks do some sanity checking like linting and formatting.
These checks must pass to merge a change.

To run these locally, you can
[install pre-commit,](https://pre-commit.com/#install)
then run `pre-commit install` inside the cloned repo. When you
commit a change, the pre-commit hooks will run automatically.
If a fix is implemented by a pre-commit hook, adding the file again
and running `git commit` a second time will pass and successfully
commit.

# Contributor License Agreement (CLA)

Triton requires that all contributors (or their corporate entity) send
a signed copy of the [Contributor License
Agreement](https://github.com/NVIDIA/triton-inference-server/blob/master/Triton-CCLA-v1.pdf)
to triton-cla@nvidia.com.
*NOTE*: Contributors with no company affiliation can fill `N/A` in the
`Corporation Name` and `Corporation Address` fields.
