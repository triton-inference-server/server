## Description: <br>
Deploy a user's ONNX file as a working Triton Inference Server endpoint, end to end. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to serve an ONNX model on NVIDIA Triton Inference Server: inspecting the ONNX graph, building the model repository, generating config.pbtxt when needed, starting and verifying the server, producing a matching Python client snippet, and debugging Triton load or inference failures on ONNX models. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [references/config-pbtxt.md](references/config-pbtxt.md) <br>
- [references/datatypes.md](references/datatypes.md) <br>
- [references/errors.md](references/errors.md) <br>
- [Triton metrics user guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md) <br>
- [Triton model repository user guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) <br>
- [Triton model configuration user guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Files] <br>
**Output Format:** [Markdown with inline bash, Python, and config.pbtxt code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Returns an output contract: model repository path, config.pbtxt contents or "Auto-completed", server command, client snippet, smoke-test result, metrics URL, and blockers] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (7 positive, 3 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox` environment). Dataset digest `sha256:4f5a3a16d08c0c6097e9f4feed77951af9e79a193496d17fec7ed7d89531219a` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.9`; evaluation date 2026-11-17. Each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 79.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 85.0% → 85.0% (±0.0 points) | 85.0% → 95.0% (+10.0 points) |
| Correctness | 66.0% → 94.0% (+28.0 points) | 64.0% → 70.0% (+6.0 points) |
| Discoverability | 73.6% — baseline ran, but no comparable score was available; uplift unavailable | 82.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 51.0% → 56.8% (+5.8 points) | 31.8% → 39.2% (+7.4 points) |
| Efficiency | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 71.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.1% → 95.7% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.9% → 90.1% (+2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Skill Version(s): <br>
eed3fb9c (source: git SHA, committed 2026-11-17) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
