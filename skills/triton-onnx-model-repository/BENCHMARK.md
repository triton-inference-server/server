# Skill Benchmark: triton-onnx-model-repository

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `triton-onnx-model-repository`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (7 positive, 3 negative)
- Dataset digest: `sha256:4f5a3a16d08c0c6097e9f4feed77951af9e79a193496d17fec7ed7d89531219a` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| triton-onnx-model-repository | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-onnx-model-repository | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-onnx-model-repository | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-onnx-model-repository | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-onnx-model-repository | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `triton-onnx-model-repository`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 74.2% — baseline ran, but no comparable score was available; uplift unavailable | 72.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 90.0% (-10.0 points) | 85.0% → 95.0% (+10.0 points) |
| Correctness | 68.0% → 86.0% (+18.0 points) | 58.0% → 80.0% (+22.0 points) |
| Discoverability | 65.7% — baseline ran, but no comparable score was available; uplift unavailable | 75.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.8% → 50.8% (+4.0 points) | 27.7% → 39.2% (+11.5 points) |
| Efficiency | 78.5% — baseline ran, but no comparable score was available; uplift unavailable | 71.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.4% → 95.0% (-1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 89.5% (+1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,686,545 | 1,819,329 | -132,784 | -7.30% | skill 10/10; base 10/10 |
| claude-code | basic-fixed-shape | 159,132 | 180,980 | -21,848 | -12.07% | skill 1/1; base 1/1 |
| claude-code | datatype-crosswalk | 35,666 | 35,392 | +274 | +0.77% | skill 1/1; base 1/1 |
| claude-code | dynamic-batching-block | 238,050 | 107,352 | +130,698 | +121.75% | skill 1/1; base 1/1 |
| claude-code | ensemble-pipeline | 117,354 | 148,051 | -30,697 | -20.73% | skill 1/1; base 1/1 |
| claude-code | gpu-placement | 201,067 | 106,859 | +94,208 | +88.16% | skill 1/1; base 1/1 |
| claude-code | multi-version-latest | 70,952 | 70,803 | +149 | +0.21% | skill 1/1; base 1/1 |
| claude-code | onnx-training | 385,552 | 840,347 | -454,795 | -54.12% | skill 1/1; base 1/1 |
| claude-code | pytorch-torchscript | 108,867 | 71,895 | +36,972 | +51.42% | skill 1/1; base 1/1 |
| claude-code | tritonclient-grpc | 185,100 | 106,560 | +78,540 | +73.70% | skill 1/1; base 1/1 |
| claude-code | tritonclient-http | 184,805 | 151,090 | +33,715 | +22.31% | skill 1/1; base 1/1 |
| codex | All cases | 1,520,046 | 1,462,429 | +57,617 | +3.94% | skill 10/10; base 10/10 |
| codex | basic-fixed-shape | 46,950 | 255,374 | -208,424 | -81.62% | skill 1/1; base 1/1 |
| codex | datatype-crosswalk | 17,453 | 17,117 | +336 | +1.96% | skill 1/1; base 1/1 |
| codex | dynamic-batching-block | 17,464 | 17,530 | -66 | -0.38% | skill 1/1; base 1/1 |
| codex | ensemble-pipeline | 51,317 | 31,308 | +20,009 | +63.91% | skill 1/1; base 1/1 |
| codex | gpu-placement | 69,572 | 41,311 | +28,261 | +68.41% | skill 1/1; base 1/1 |
| codex | multi-version-latest | 34,270 | 44,705 | -10,435 | -23.34% | skill 1/1; base 1/1 |
| codex | onnx-training | 829,512 | 819,155 | +10,357 | +1.26% | skill 1/1; base 1/1 |
| codex | pytorch-torchscript | 31,687 | 27,332 | +4,355 | +15.93% | skill 1/1; base 1/1 |
| codex | tritonclient-grpc | 48,637 | 27,182 | +21,455 | +78.93% | skill 1/1; base 1/1 |
| codex | tritonclient-http | 373,184 | 181,415 | +191,769 | +105.71% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,206,591 | 3,281,758 | -75,167 | -2.29% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in config-pbtxt.md (`skills/triton-onnx-model-repository/SKILL.md`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:40`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:72`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:157`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:222`)
- 4 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
