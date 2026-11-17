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
- Tasks: 12 evaluation tasks (12 positive)
- Dataset digest: `sha256:e8dea4a558c0f17d152b0015405a05b99067398767dc1cdc646d8e1dda261dc3` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 95.8% (-4.2 points) | 91.7% → 87.5% (-4.2 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 95.0% → 95.0% (±0.0 points) |
| Discoverability | 72.1% — baseline ran, but no comparable score was available; uplift unavailable | 78.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 72.2% → 96.5% (+24.3 points) | 75.0% → 79.9% (+4.9 points) |
| Efficiency | 83.0% — baseline ran, but no comparable score was available; uplift unavailable | 69.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.5% → 96.0% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.3% → 88.9% (+2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,874,661 | 1,518,159 | +356,502 | +23.48% | skill 12/12; base 12/12 |
| claude-code | deploy-pt2-gpu | 248,254 | 186,501 | +61,753 | +33.11% | skill 1/1; base 1/1 |
| claude-code | deploy-torchscript-batched | 242,931 | 185,053 | +57,878 | +31.28% | skill 1/1; base 1/1 |
| claude-code | error-aoti-device-mismatch | 115,823 | 71,905 | +43,918 | +61.08% | skill 1/1; base 1/1 |
| claude-code | error-missing-file-ts | 154,114 | 71,688 | +82,426 | +114.98% | skill 1/1; base 1/1 |
| claude-code | naming-pt2-ordinals | 115,374 | 71,778 | +43,596 | +60.74% | skill 1/1; base 1/1 |
| claude-code | onnx-file-wrong-skill | 36,297 | 36,210 | +87 | +0.24% | skill 1/1; base 1/1 |
| claude-code | perf-warmup | 153,381 | 106,427 | +46,954 | +44.12% | skill 1/1; base 1/1 |
| claude-code | pt2-dynamic-batch | 154,798 | 107,053 | +47,745 | +44.60% | skill 1/1; base 1/1 |
| claude-code | seq-batch-torchscript | 364,454 | 354,233 | +10,221 | +2.89% | skill 1/1; base 1/1 |
| claude-code | torchscript-vs-pt2 | 35,425 | 35,171 | +254 | +0.72% | skill 1/1; base 1/1 |
| claude-code | torchserve-not-triton | 145,575 | 107,199 | +38,376 | +35.80% | skill 1/1; base 1/1 |
| claude-code | training-not-inference | 108,235 | 184,941 | -76,706 | -41.48% | skill 1/1; base 1/1 |
| codex | All cases | 1,027,995 | 1,330,756 | -302,761 | -22.75% | skill 12/12; base 12/12 |
| codex | deploy-pt2-gpu | 68,588 | 53,335 | +15,253 | +28.60% | skill 1/1; base 1/1 |
| codex | deploy-torchscript-batched | 216,231 | 407,976 | -191,745 | -47.00% | skill 1/1; base 1/1 |
| codex | error-aoti-device-mismatch | 50,726 | 66,561 | -15,835 | -23.79% | skill 1/1; base 1/1 |
| codex | error-missing-file-ts | 33,574 | 17,715 | +15,859 | +89.52% | skill 1/1; base 1/1 |
| codex | naming-pt2-ordinals | 50,091 | 30,058 | +20,033 | +66.65% | skill 1/1; base 1/1 |
| codex | onnx-file-wrong-skill | 22,624 | 26,810 | -4,186 | -15.61% | skill 1/1; base 1/1 |
| codex | perf-warmup | 59,281 | 35,457 | +23,824 | +67.19% | skill 1/1; base 1/1 |
| codex | pt2-dynamic-batch | 37,778 | 31,397 | +6,381 | +20.32% | skill 1/1; base 1/1 |
| codex | seq-batch-torchscript | 342,787 | 567,405 | -224,618 | -39.59% | skill 1/1; base 1/1 |
| codex | torchscript-vs-pt2 | 41,921 | 21,998 | +19,923 | +90.57% | skill 1/1; base 1/1 |
| codex | torchserve-not-triton | 62,761 | 31,298 | +31,463 | +100.53% | skill 1/1; base 1/1 |
| codex | training-not-inference | 41,633 | 40,746 | +887 | +2.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,902,656 | 2,848,915 | +53,741 | +1.89% | skill 24/24; base 24/24 |

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
