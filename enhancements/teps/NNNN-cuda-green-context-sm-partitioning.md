# Allow Triton Model Instance Groups to Select SM Partitions with CUDA Green Contexts

| Proposal information | Details |
|---|---|
| Status | Draft |
| Authors | TypeFloat (Zekun Liu) |
| Category | Architecture |
| Sponsor | TBD |
| Proposed reviewers | Maintainers of the Core model instance lifecycle, model configuration, and TensorRT backend. Input from a backend with a different approach to CUDA stream management would also help evaluate the public API. |
| Review date | TBD |
| Tracking issue | [triton-inference-server/server#9083](https://github.com/triton-inference-server/server/issues/9083) |

## Summary

We propose adding CUDA Green Context support to Triton Inference Server so that latency-sensitive models can use dedicated sets of streaming multiprocessors (SMs) when sharing a GPU with other models. We have implemented a TensorRT backend prototype and evaluated it under two mixed workloads. We would like to agree with maintainers on the user-visible behavior, the responsibilities of Core and backends, and the scope of an initial upstream implementation.

## Motivation

CUDA 13.1 added Green Context support to the Runtime API. Within a single process, an application can partition a GPU’s SMs into disjoint sets, create CUDA streams associated with each set, and submit work through those streams. For example, an application can run latency-sensitive inference on one set of SMs while a background workload processes large batches on another. This gives the application control over which workloads compete for the same compute resources. [NVIDIA’s introduction to Green Contexts](https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/)

Developers serving models through Triton currently have no built-in way to use this capability to run different models on separate sets of SMs within the same GPU. We would like to make this possible while preserving Triton’s single-process, multi-model deployment model.

We evaluated a prototype on one NVIDIA GeForce RTX 5090 D v2 with 170 SMs, CUDA 13.1, and TensorRT 11.4. Each model had one instance. Within each experiment, the shared-GPU and partitioned runs used the same Triton binary, TensorRT plans, and request arrival schedules. We validated the inference outputs in both configurations.

### Experiment 1: Can a foreground model meet a 1 ms latency budget while a background model is overloaded?

The foreground model was latency-sensitive and received 1,000 requests/s throughout the experiment. The background model prioritized throughput and received more requests than it could complete. We measured how often foreground requests exceeded 1 ms and how much background work completed under the same offered load.

| GPU allocation | Foreground P99 | Foreground requests over 1 ms (of 10,800) | Completed background requests/s |
|---|---:|---:|---:|
| Both models share all 170 SMs | 1.245 ms | 2,163 | ~197 |
| 8 SMs for the foreground model, 162 for the background model | 0.197 ms | 0 | ~197 |

The background model accumulated a backlog in both runs. Assigning 8 SMs to the foreground model brought every measured foreground request within budget, while observed background throughput remained approximately the same.

### Experiment 2: Can partitioning improve P99 for two compute-intensive models that interfere with each other?

Models A and B both had latency targets. Each received regular requests and periodic large batches on a predefined schedule. We staggered the large batches so that A processed a large batch while B continued to receive regular requests, and vice versa. We replayed the same arrival schedule under both GPU configurations and measured P99 separately for each model.

| GPU allocation | Model A P99 | Model B P99 |
|---|---:|---:|
| A and B share all 170 SMs | 17.549 ms | 14.335 ms |
| 80 SMs for A, 90 for B | 0.844 ms | 0.945 ms |

Under this schedule, partitioning brought P99 below 1 ms for both models. In a separate control with fixed input sizes and sufficiently staggered requests, partitioning increased P99 for both models. These results support keeping whole-GPU sharing as the default and allowing operators to opt in based on their workload and latency targets.

### Goals and scope

The initial scope is single-GPU TensorRT inference within one Triton process. An instance group can select an SM partition, and multiple instances can share a partition. The prototype supports direct inference and CUDA Graph capture and replay.

The isolation applies to compute submitted through streams associated with disjoint SM sets. Work on ordinary streams or in other processes can still compete for those SMs. GPU memory and memory bandwidth also remain shared. Runtime repartitioning, automatic partition sizing, and inference that coordinates multiple devices are outside the initial scope.

## Proposal

We propose three capabilities:

1. **Define named partitions per GPU.** At server startup, the operator defines an SM layout for each GPU that needs partitioning. Triton Core manages that layout for all models on the device.
2. **Select a partition per instance group.** A model’s `instance_group` can reference a partition by name. Instances in the group share that partition, and groups from different models can select the same partition. Groups that omit the field retain their existing execution behavior.
3. **Run inference on the selected partition.** A participating backend uses the Green Context bound to the model instance when submitting compute. TensorRT is the first backend in the prototype. Other backends can add support according to how their frameworks manage execution.

### Configuration example

In the first experiment, GPU 0 has an `interactive` partition with 8 SMs and a `background` partition with the remaining SMs. In the prototype syntax, a positive count requests exactly that many SMs. A count of `0` assigns the remainder to one partition per GPU. If the hardware cannot satisfy the requested layout, server initialization fails rather than silently rounding the allocation.

```sh
tritonserver --model-repository=/models \
  --green-context-partition=0:interactive:8 \
  --green-context-partition=0:background:0
```

The foreground model selects `interactive` in its model configuration:

```protobuf
instance_group [{
  kind: KIND_GPU
  gpus: [0]
  count: 1
  green_context_partition: "interactive"
}]
```

The background model selects `background` in its own instance group. Increasing the foreground group’s `count` creates more instances sharing the same 8 SMs. It does not allocate another 8 SMs per instance.

### Execution guarantee and failure behavior

An invalid device layout should fail server initialization. An explicit reference to an unknown partition, or a request to use a partition in an unsupported execution mode, should fail model loading. The prototype uses the CUDA 13.1 or later Runtime API. Existing backends should continue to serve models that do not request partitioning when running against the updated Core.

**For the initial TensorRT implementation, we propose a guarantee covering the instance’s primary compute stream.** The backend submits direct inference on a stream associated with the selected partition and uses that stream for CUDA Graph capture and replay. We have exercised both paths on a single GPU. Framework-managed auxiliary streams and work submitted independently by plugins require separate verification. If the intended contract is that *all* compute for an inference request must stay within its partition, the initial implementation would also need to verify those paths or restrict the model configurations eligible for partitioning.

### Prototype changes and design trade-offs

| Repository | Prototype change | Design rationale |
|---|---|---|
| `server` | A repeatable startup option specifies partition names and SM counts per device. A server options API passes the layout to Core and is also available to applications embedding Triton. | Command-line and embedded deployments can use the same layout semantics. Operators define the layout once, and models reference it. |
| `common` | An optional `green_context_partition` field on `instance_group` lets different groups of the same model select different partitions. | GPU placement, instance count, and partition selection can be expressed together. Increasing the instance count does not allocate additional SMs. |
| `core` | Core creates the partitions at startup and owns their Green Contexts. At instance load, it resolves the partition using the instance’s actual GPU and partition name, then exposes a borrowed handle to the backend. | The layout has a lifetime independent of any one model. If two models share a partition, unloading one should allow the other to continue serving. |
| `tensorrt_backend` | The backend creates the instance’s compute stream on the selected context and uses it for direct execution and graph capture and replay. Under device-blocking execution, the prototype preserves the scheduled instance when the same model has different bindings on a GPU. It retains the existing rotation behavior when bindings are identical. | If Triton schedules an instance bound to partition A but the backend rotates execution to an instance bound to partition B, the request runs on the wrong partition even though both instances have correctly configured streams. |

**Model configuration changes.** The TensorRT backend chooses its instance arbitration strategy when the model is created. In the prototype, Core detects partitioned TensorRT models during reload checks. If either the old or new configuration uses partitioning, an instance group change triggers a full model reload, including a change to instance count alone. This prevents stale arbitration behavior after bindings are added, removed, or changed. The trade-off is a broader reload and TensorRT-specific logic in Core. An upstream alternative would let a backend declare which instance group changes require rebuilding model state.

**Public backend API.** The prototype exposes `TRITONBACKEND_ModelInstanceGreenContext`, which lets the backend borrow a Core-owned context and create a stream appropriate for its framework. Another option is for Core to create a stream for a given instance and priority, provided the framework accepts an externally created stream. Input from backends with different stream ownership models would help determine which interface is more useful.

**Verifying that a binding is honored.** Core can reject a backend that never queries an explicitly requested binding. It cannot infer from a successful query that the backend subsequently used the partition: a backend could retrieve the handle and still submit work on an ordinary stream. Acceptance tests should detect that case. They should also distinguish two partitions with equal SM counts, where checking the count alone would miss a request that ran on the wrong partition. Exact API signatures and internal cleanup ordering can be addressed during implementation review.

### Implementation sequence and questions for reviewers

We propose agreeing on the user-visible binding behavior and backend execution guarantee first, then submitting implementation PRs to the four repositories above. We would appreciate feedback on three decisions:

1. Does a per-device layout owned by Core, with named selection at instance group granularity and a load error when an explicit binding cannot be honored, fit Triton’s architecture?
2. Should backend support guarantee placement of the primary compute stream, or all compute for an inference request, including auxiliary streams and plugins? How should a backend declare or reject execution paths it cannot control?
3. Which framework constraints should guide the choice between a borrowed context and a Core-created stream? Which maintainers should review the public API, and would a maintainer or code owner be willing to sponsor this TEP?

Perf Analyzer could measure latency and throughput for a given layout, and Model Analyzer could help compare candidate layouts. We propose treating this tooling as a follow-up once the configuration and execution contract are stable.

## Alternate Solutions

**Manage partitions entirely in the TensorRT backend.** This would cover both workloads evaluated here and involve fewer components initially. We favor Core ownership because other backends may eventually share the same GPU with TensorRT. A Core-owned layout lets them reference the same named partitions without independently coordinating SM allocation.

**Continue sharing the whole GPU.** When requests are sufficiently staggered, each model can use all available SMs while the other is idle. In our control experiment, both models had better P99 latency under whole-GPU sharing. This is why we propose preserving it as the default and making partitioning an explicit deployment choice.
