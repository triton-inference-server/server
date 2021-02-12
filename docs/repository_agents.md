<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# Repository Agent

A *repository agent* extends Triton with new functionality that
operates when a model is loaded or unloaded. You can introduce your
own code to perform authentication, decryption, conversion, or similar
operations when a model is loaded.

**BETA: The repository agent API is beta quality and is subject to
non-backward-compatible changes for one or more releases.**

A repository agent comunicates with Triton using the [repository agent
API](https://github.com/triton-inference-server/core/tree/main/include/triton/core/tritonrepoagent.h). The
[checksum_repository_agent GitHub
repo](https://github.com/triton-inference-server/checksum_repository_agent)
provides an example repository agent that verifies file checksums
before loading a model.

## Using a Repository Agent

A model can use one or more repository agents by specifying them in
the *ModelRepositoryAgents* section of the [model
configuration](model_configuration.md). Each repository agent can have
parameters specific to that agent that are specified in the model
configuration to control the behavior of the agent. To understand the
parameters available for a given agent consult the documentation for
that agent.

Multiple agents may be specified for the same model and they will be
invoked in order when a model is loaded or unloaded. The following
example model configuration contents shows how two agents, "agent0"
and "agent1", are specified so that they are invoked in that order
with the given parameters.

```
model_repository_agents
{
  agents [
    {
      name: "agent0",
      parameters [
        {
          key: "key0",
          value: "value0"
        },
        {
          key: "key1",
          value: "value1"
        }
      ]
    },
    {
      name: "agent1",
      parameters [
        {
          key: "keyx",
          value: "valuex"
        }
      ]
    }
  ]
}
```

## Implementing a Repository Agent

A repository agent must be implemented as a shared library and the
name of the shared library must be
*libtritonrepoagent_\<repo-agent-name\>.so*. The shared library should
hide all symbols except those needed by the repository agent API. See
the [checksum example's
CMakeList.txt](https://github.com/triton-inference-server/checksum_repository_agent/blob/main/CMakeLists.txt)
for an example of how to use an ldscript to expose only the necessary
symbols.

The shared library will be dynamically loaded by Triton when it is
needed. For a repository agent called *A*, the shared library must be
installed as \<repository_agent_directory\>/A/libtritonrepoagent_A.so.
Where \<repository_agent_directory\> is by default
/opt/tritonserver/repoagents.  The --repoagent-directory flag can be
used to override the default.

Your repository agent must implement the repository agent API as
documented in
[tritonrepoagent.h](https://github.com/triton-inference-server/core/tree/main/include/triton/core/tritonrepoagent.h).

Triton follows these steps when loading a model:

* Load the model's configuration file (config.pbtxt) and extract the
  *ModelRepositoryAgents* settings. Even if a repository agent
  modifies the config.pbtxt file, the repository agent settings from
  the initial config.pbtxt file are used for the entire loading
  process.

* For each repository agent specified:

  * Initialize the corresponding repository agent, loading the shared
    library if necessary. Model loading fails if the shared library is
    not available or if initialization fails.

  * Invoke the repository agent's *TRITONREPOAGENT_ModelAction*
    function with action TRITONREPOAGENT_ACTION_LOAD. As input the
    agent can access the model's repository as either a cloud storage
    location or a local filesystem location.

  * The repository agent can return *success* to indicate that no
    changes where made to the repository, can return *failure* to
    indicate that the model load should fail, or can create a new
    repository for the model (for example, by decrypting the input
    repository) and return *success* to indicate that the new
    repository should be used.

  * If the agent returns *success* Triton continues to the next
    agent. If the agent returns *failure*, Triton skips invocation of
    any additional agents.

* If all agents returned *success*, Triton attempts to load the model
  using the final model repository.

* For each repository agent that was invoked with
  TRITONREPOAGENT_ACTION_LOAD, in reverse order:

  * Triton invokes the repository agent's
    *TRITONREPOAGENT_ModelAction* function with action
    TRITONREPOAGENT_ACTION_LOAD_COMPLETE if the model loaded
    successfully or TRITONREPOAGENT_ACTION_LOAD_FAIL if the model
    failed to load.

Triton follows these steps when unloading a model:

* Triton uses the repository agent settings from the initial
  config.pbtxt file, even if during loading one or more agents
  modified its contents.

* For each repository agent that was invoked with
  TRITONREPOAGENT_ACTION_LOAD, in the same order:

  * Triton invokes the repository agent's
    *TRITONREPOAGENT_ModelAction* function with action
    TRITONREPOAGENT_ACTION_UNLOAD.

* Triton unloads the model.

* For each repository agent that was invoked with
  TRITONREPOAGENT_ACTION_UNLOAD, in reverse order:

  * Triton invokes the repository agent's
    *TRITONREPOAGENT_ModelAction* function with action
    TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE.
