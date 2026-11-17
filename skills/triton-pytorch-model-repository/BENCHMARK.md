# Skill Benchmark: triton-pytorch-model-repository

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `triton-pytorch-model-repository`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 12 evaluation tasks (9 positive, 3 negative)
- Dataset digest: `sha256:6590ed5a89d7c12a607bc2f8305ff42c176a8f43d043fd07c29d22c4781e2e9d` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| triton-pytorch-model-repository | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-pytorch-model-repository | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-pytorch-model-repository | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-pytorch-model-repository | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| triton-pytorch-model-repository | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `triton-pytorch-model-repository`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 91.7% (+4.2 points) |
| Correctness | 83.3% → 98.3% (+15.0 points) | 96.7% → 95.0% (-1.7 points) |
| Discoverability | 87.2% — baseline ran, but no comparable score was available; uplift unavailable | 90.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 57.6% → 91.0% (+33.4 points) | 84.7% → 84.0% (-0.7 points) |
| Efficiency | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 71.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 96.2% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.0% → 88.9% (+1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,827,205 | 1,326,260 | +500,945 | +37.77% | skill 12/12; base 12/12 |
| claude-code | deploy-pt2-gpu | 204,000 | 107,253 | +96,747 | +90.20% | skill 1/1; base 1/1 |
| claude-code | deploy-torchscript-batched | 196,151 | 187,935 | +8,216 | +4.37% | skill 1/1; base 1/1 |
| claude-code | error-aoti-device-mismatch | 115,642 | 71,677 | +43,965 | +61.34% | skill 1/1; base 1/1 |
| claude-code | error-missing-file-ts | 153,220 | 71,401 | +81,819 | +114.59% | skill 1/1; base 1/1 |
| claude-code | naming-pt2-ordinals | 115,406 | 71,407 | +43,999 | +61.62% | skill 1/1; base 1/1 |
| claude-code | onnx-file-wrong-skill | 71,824 | 36,025 | +35,799 | +99.37% | skill 1/1; base 1/1 |
| claude-code | perf-warmup | 191,072 | 71,369 | +119,703 | +167.72% | skill 1/1; base 1/1 |
| claude-code | pt2-dynamic-batch | 154,901 | 70,701 | +84,200 | +119.09% | skill 1/1; base 1/1 |
| claude-code | seq-batch-torchscript | 292,385 | 350,190 | -57,805 | -16.51% | skill 1/1; base 1/1 |
| claude-code | torchscript-vs-pt2 | 35,745 | 35,338 | +407 | +1.15% | skill 1/1; base 1/1 |
| claude-code | torchserve-not-triton | 108,633 | 106,825 | +1,808 | +1.69% | skill 1/1; base 1/1 |
| claude-code | training-not-inference | 188,226 | 146,139 | +42,087 | +28.80% | skill 1/1; base 1/1 |
| codex | All cases | 960,712 | 1,603,721 | -643,009 | -40.09% | skill 12/12; base 12/12 |
| codex | deploy-pt2-gpu | 67,612 | 66,425 | +1,187 | +1.79% | skill 1/1; base 1/1 |
| codex | deploy-torchscript-batched | 196,009 | 210,247 | -14,238 | -6.77% | skill 1/1; base 1/1 |
| codex | error-aoti-device-mismatch | 51,098 | 66,580 | -15,482 | -23.25% | skill 1/1; base 1/1 |
| codex | error-missing-file-ts | 33,361 | 17,802 | +15,559 | +87.40% | skill 1/1; base 1/1 |
| codex | naming-pt2-ordinals | 49,883 | 25,573 | +24,310 | +95.06% | skill 1/1; base 1/1 |
| codex | onnx-file-wrong-skill | 30,191 | 22,624 | +7,567 | +33.45% | skill 1/1; base 1/1 |
| codex | perf-warmup | 52,081 | 31,241 | +20,840 | +66.71% | skill 1/1; base 1/1 |
| codex | pt2-dynamic-batch | 37,740 | 31,265 | +6,475 | +20.71% | skill 1/1; base 1/1 |
| codex | seq-batch-torchscript | 298,415 | 1,019,544 | -721,129 | -70.73% | skill 1/1; base 1/1 |
| codex | torchscript-vs-pt2 | 37,595 | 22,014 | +15,581 | +70.78% | skill 1/1; base 1/1 |
| codex | torchserve-not-triton | 49,925 | 49,233 | +692 | +1.41% | skill 1/1; base 1/1 |
| codex | training-not-inference | 56,802 | 41,173 | +15,629 | +37.96% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,787,917 | 2,929,981 | -142,064 | -4.85% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/config-pbtxt.md:
  "## Platform Form" in SKILL.md (lines 20-29)
  vs "## Platform form" in references/config-pbtxt.md (lines 10-19) (`SKILL.md:20`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in config-pbtxt.md (`skills/triton-pytorch-model-repository/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/triton-pytorch-model-repository/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/triton-pytorch-model-repository/SKILL.md`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -s -X POST -H "Content-Type: application/json" \
  -d  (`SKILL.md:113`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
