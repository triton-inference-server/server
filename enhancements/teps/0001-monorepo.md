<!--
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
# Triton Monolithic Repository

**Status**: Draft <!-- \[Draft | Under Review | Approved | Replaced | Deferred | Rejected\] -->

**Authors**: J Wyman <jwyman@nvidia.com> ([whoisj](https://github.com/whoisj)) <!-- \[Name/Team\] -->

**Category**: Process <!-- \[Architecture | Process | Guidelines\] -->

**Replaces**: _n/a_ <!-- \[Link of previous proposal if applicable\] -->

**Replaced By**: _n/a_ <!-- \[Link of previous proposal if applicable\] -->

**Sponsor**: J Wyman <jwyman@nvidia.com> ([whoisj](https://github.com/whoisj)) <!-- \[Name of code owner or maintainer to shepherd process\] -->

**Required Reviewers**:
<!-- \[Names of technical leads that are required for acceptance\] -->
* Mudit Aggarwal <mudita@nvidia.com> ([mudit-eng](https://github.com/mudit-eng))
* Misha Chornyi <mchornyi@nvidia.com> ([mc-nv](https://github.com/mc-nv))

**Requested Reviewers**:
<!-- \[Names of additional contributors and/or maintainers requested\] -->
* Rini Gupta <rinig@nvidia.com> ([nv-rinig](https://github.com/nv-rinig))
* Vinya Kestur <vinyal@nvidia.com> ([vinya567](https://github.com/vinya567))
* Yingge He <yinggeh@nvidia.com> ([yinggeh](https://github.com/yinggeh))

**Review Date**: _tbd_ <!-- \[Date for review\] -->

**Pull Request**: _tbd_ <!-- \[Link to Pull Request of the Proposal itself\] -->

**Implementation PR / Tracking Issue**: _tbd_ <!-- \[Link to Pull Request or Tracking Issue for Implementation\] -->

## Summary

Reduce the number of Triton repositories by merging the
  [backend](https://github.com/triton-inference-server/backend),
  [common](https://github.com/triton-inference-server/common),
  [core](https://github.com/triton-inference-server/core), and
  [python_backend](https://github.com/triton-inference-server/python_backend)
  repositories into the [server](https://github.com/triton-inference-server/server) repository.
Preserving commit history by renaming server's `main` branch to `legacy-main` and archiving the other repositories.

## Motivation

The current solution requires multiple repositories to cloned and checked out to the correct branch in order to build/test.
This causes a number of problems:

* Overly complicated for developers not familiar with the solution to effectively work with Triton.
  * Discourages third-party contributions to the open-source project.
  * Discourages third-patty adoption of Triton when they require customization or modification of the source code or build procedures.
  * Requires manual management and coordination or cross-repository changes that depend on each other.

* Increases the complexity of Triton's build system.
  * Build system requires the capacity to understand multiple Git repositories and associated branches.
  * Negatively impacts build caching solutions and technology.
  * Increases chances for human error.
  * Requires complex instructions for CI systems to correctly handle build and test automation.

* Increases the complexity of managing release branches.
  * Requires branching multiple repositories in unison.
  * Requires version information to manually propagated across multiple repositories.

### Goals

* Reduce the "total cost of ownership" for Triton Inference Server.

* Reduce barrier to third-party adoption and contribution.

* Reduce build complexity.

* Avoid code that always branches and builds together being split into multiple repositories and branches.

#### Non Goals

* Reduce build times.

* Reduce test times.

* Rename the server repository.

* Improve the underlying build system and methodologies.

* Improve testing system and methodologies.

## Requirements

### REQ 01  Retain All Code, Functionality, and Capabilities

Code from all merged repositories **MUST** be retained.
All functionality, capabilities, and features of the existing Triton Inference Server **MUST** be retained.
Triton core and backend C-ABI and Python API compatibility **MUST** be retained.

### REQ 02  Remain GitHub Compatible

The merged repository **MUST** remain compatible w/ GitHub.
Specifically, support for issue templates, workflows, etc. **MUST** be retained.

### REQ 03  Remain Cmake Buildable

The merged repository **MUST** be buildable using Cmake.

This likely requires changes to the existing Cmake files removing steps to download external repositories (now merged).
The final result **SHOULD** support building either target: Triton Server or Triton Core.

### REQ 04  Remain Compatible w/ NVIDIA Internal CI Labs

The merged repository **MUST NOT** break the ability to build and test Triton Server changes using NVIDIA's internal CI labs.

## Proposal

Merge the
  [backend](https://github.com/triton-inference-server/backend),
  [common](https://github.com/triton-inference-server/common),
  [core](https://github.com/triton-inference-server/core),
  [python_backend](https://github.com/triton-inference-server/python_backend), and
  [server](https://github.com/triton-inference-server/server)
  repositories into a single, monolithic repository.

The merged repositories would be merged into a new `main` branch with a new base commit.
The existing `main` branch would be renamed to `legacy-main` and preserved in order to retain commit history.

The initial commit of the new `main` branch would include the basic setup of `.gitattrbutes` and `.gitignore` along with a commit message explaining what is happening with the repository.
Ideally, this message would include information along the lines of:

> Initial Commit for Triton Inference Server v2
>
> Restructuring Triton Inference Server multi-repository into a monolithic repository to
> * Reduce the “total cost of ownership” for Triton Inference Server.
> * Reduce barrier to third-party adoption and contribution.
> * Reduce build complexity.
> * Keep code that builds and branches together in the same repository.
>
> This first commit clears the all content from main except
> * .github/
> * .dockerignore
> * .gitattributes
> * .gitignore
> * .pre-commit-config.yaml
>
> Previous `main` branch has been renamed to `legacy-main` to retain history.
> Other branches in this repository are unaffected.

The second commit in the repository would have a massage like:

> Merger of Multiple Repositories into Monolithic Repository
>
> Content comes from the following sources:
> * https://github.com/triton-inference-server/server@legacy-main:<commit-sha>
> * https://github.com/triton-inference-server/core@main:<commit-sha>
> * https://github.com/triton-inference-server/common@main:<commit-sha>
> * https://github.com/triton-inference-server/backend@main:<commit-sha>
> * https://github.com/triton-inference-server/python_backend@main:<commit-sha>

If there is concern about attribution, then we can create a script that creates commits and properly attributes changes to specific contributors.
This would impose a onetime cost, but provide clarity when understanding who contributed code line-by-line.

The table below is the proposed structure for the monolithic repository.
Each path in the repository has a corresponding repository + path combination explaining where each folder and/or file is sourced from.

| Destination                                  | Source (**repository**:`path`)                                 |   Type   |
| :------------------------------------------- | :------------------------------------------------------------- | :------: |
| `.devcontainer/`                             | **python_backend**:`/.devcontainer/`                           | _folder_ |
| `.github/`                                   | **server**:`/.github/`                                         | _folder_ |
| `cmake/define.cuda_architectures.cmake`      | **backend**:`/cmake/define.cuda_architectures.cmake`           |  _file_  |
| `cmake/TritonBackendConfig.cmake.in`         | **backend**:`/cmake/TritonBackendConfig.cmake.in`              |  _file_  |
| `cmake/TritonCommonConfig.cmake.in`          | **common**:`/cmake/TritonCommonConfig.cmake.in`                |  _file_  |
| `cmake/TritonCoreConfig.cmake.in`            | **core**:`/cmake/TritonCoreConfig.cmake.in`                    |  _file_  |
| `cmake/TritonPythonBackendConfig.cmake.in`   | **python_backend**:`/cmake/TritonPythonBackendConfig.cmake.in` |  _file_  |
| `deploy/`                                    | **server**:`/deploy/`                                          | _folder_ |
| `docker/`                                    | **server**:`/docker/`                                          | _folder_ |
| `docs/`                                      | **server**:`/docs/`                                            | _folder_ |
| `docs/backend/`                              | **backend**:`/docs/`                                           | _folder_ |
| `enhancements/`                              | **server**:`/enhancements/`                                    | _folder_ |
| `qa/`                                        | **server**:`/qa`                                               | _folder_ |
| `src/CMakeLists.txt`                         | _New root cmake file for building all repositories._           |  _file_  |
| `src/backend/examples/`                      | **backend**:`/examples/`                                       | _folder_ |
| `src/backend/include/`                       | **backend**:`/include/`                                        | _folder_ |
| `src/backend/src/`                           | **backend**:`/src/`                                            | _folder_ |
| `src/backend/CmakeLists.txt`                 | **backend**:`/CMakeLists.txt`                                  |  _file_  |
| `src/backend/pyproject.toml`                 | **backend**:`/pyproject.toml`                                  |  _file_  |
| `src/common/include/`                        | **common**:`/include/`                                         | _folder_ |
| `src/common/protobuf/`                       | **common**:`/protobuf/`                                        | _folder_ |
| `src/common/src/`                            | **common**:`/src/`                                             | _folder_ |
| `src/common/CmakeLists.txt`                  | **common**:`/CMakeLists.txt`                                   |  _file_  |
| `src/common/pyproject.toml`                  | **common**:`/pyproject.toml`                                   |  _file_  |
| `src/core/include/`                          | **core**:`/include/`                                           | _folder_ |
| `src/core/python/`                           | **core**:`/python/`                                            | _folder_ |
| `src/core/src/`                              | **core**:`/src/`                                               | _folder_ |
| `src/core/CmakeLists.txt`                    | **core**:`/CMakeLists.txt`                                     |  _file_  |
| `src/core/pyproject.toml`                    | **core**:`/pyproject.toml`                                     |  _file_  |
| `src/python_backend/examples/`               | **python_backend**:`/examples/`                                | _folder_ |
| `src/python_backend/inferentia/`             | **python_backend**:`/inferentia/`                              | _folder_ |
| `src/python_backend/src/`                    | **python_backend**:`/src/`                                     | _folder_ |
| `src/python_backend/CmakeLists.txt`          | **python_backend**:`/CMakeLists.txt`                           |  _file_  |
| `src/python_backend/pyproject.toml`          | **python_backend**:`/pyproject.toml`                           |  _file_  |
| `src/server/python/`                         | **server**:`/python/`                                          | _folder_ |
| `src/server/src/`                            | **server**:`/src/`                                             | _folder_ |
| `src/server/tools/`                          | **server**:`/tools/`                                           | _folder_ |
| `src/server/CmakeLists.txt`                  | **server**:`/CMakeLists.txt`                                   |  _file_  |
| `.CITATION.cff`                              | **server**:`/.CITATION.cff`                                    |  _file_  |
| `.dockerignore`                              | **server**:`/.dockerignore`                                    |  _file_  |
| `.gitattributes`                             | _New attribute handling file for Git._                         |  _file_  |
| `.gitignore`                                 | **server**:`/.gitignore`                                       |  _file_  |
| `.pre-commit-config.yaml`                    | **server**:`/.pre-commit-config.yaml`                          |  _file_  |
| `CONTRIBUTING.md`                            | **server**:`/CONTRIBUTING.md`                                  |  _file_  |
| `Dockerfile.QA`                              | **server**:`/Dockerfile.QA`                                    |  _file_  |
| `Dockerfile.SDK`                             | **server**:`/Dockerfile.SDK`                                   |  _file_  |
| `LICENSE`                                    | **server**:`/LICENSE`                                          |  _file_  |
| `NVIDIA_Deep_Learning_Container_License.pdf` | **server**:`/NVIDIA_Deep_Learning_Container_License.pdf`       |  _file_  |
| `README.md`                                  | **server**:`/README.md`                                        |  _file_  |
| `SECURITY.md`                                | **server**:`/SECURITY.md`                                      |  _file_  |
| `TRITON_VERSION`                             | **server**:`/TRITON_VERSION`                                   |  _file_  |
| `Triton-CCLA-v1.pdf`                         | **server**:`/Triton-CCLA-v1.pdf`                               |  _file_  |
| `build.py`                                   | **server**:`/build.py`                                         |  _file_  |
| `compose.py`                                 | **server**:`/compose.py`                                       |  _file_  |

## Alternate Solutions

### Alt 01  Same as Above Plus `third_party`

Include the `third_party` repository in the set of repositories merged into the server repository.

**Pros:**

* Reduced complexity.
* Minor benefits to build system.

**Cons:**

* The `third_party` repository is sequestered because it contains non-NVIDIA IP.
  * Merging it could introduce complexities w/ regards to licenses and IP management.

**Reason Rejected:**

* See Cons above.

* The `third_party` repository is rarely updated; maintenance costs are already low.

### Alt 02  Use Submodules

Use Git submodules to connect the server repository to the other repositories instead of merging them.

**Pros:**

* Minimal disruption.
  * No impact on pending pull-requests.
  * Easiest for third-parties, that maintain forks or Triton, to ingest.

* Least impact on change history.

**Cons:**

* Significantly less positive impact on build system complexity.

* No improvement to branch management.
  * Multiple release branches per Triton release.
  * Manual management and coordination or cross-repository changes.
  * Complexity discouraging third-party adoption and contributions.

**Reason Rejected:**

* Not enough positive impact.

* Third-party discouragement.

### Alt 03  Do Nothing

**Pros:**

* No effort cost.

* No workflow disruption.

* Least risk involved.

**Cons:**

* Relatively high cost of development and maintenance.

* Discourages third-party adoption and contributions.

* No improvements for build and/or test systems.

**Reason Rejected:**

* The status quo isn't good enough.

* Ownership costs are rising.

* Third-party discouragement.
