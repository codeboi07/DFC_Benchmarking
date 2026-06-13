# Tau3/Tau2 DFC Benchmark Runner With AWS Bedrock Claude Sonnet

This repo contains a runner at `scripts/tau3_dfc_benchmark.py`.

It targets the current tau-three codebase in `sierra-research/tau2-bench`. The Sierra
`tau-3-bench` page is the March 18, 2026 `τ-Voice` release; its paper says the
implementation lives in `sierra-research/tau2-bench`. This runner uses the text
`tau2 run` path because AWS Bedrock Claude Sonnet is a text/tool-calling model, not
one of the audio-native realtime providers used for full-duplex voice.

## What The Script Does

1. Clones or reuses a tau2-bench checkout.
2. Installs tau2 dependencies with `uv`.
3. Generates an auditable DFC sink map for all selected domains:
   `generated/tau3/generated_tau3_dfc_policies.json`.
4. Installs a `dfc_llm_agent` adapter into the tau2 checkout.
5. Runs baseline tau2 benchmarks with `llm_agent`.
6. Optionally runs DFC benchmarks with `dfc_llm_agent`.
7. Writes side-by-side utility analysis:
   `generated/tau3/analysis/summary.csv` and `by_task.csv`.

Default benchmark domains include the three Sierra `τ-Voice` domains plus the
separate Sierra `τ-Knowledge` banking domain:

- `retail`
- `airline`
- `telecom`
- `banking_knowledge`

The `τ-Voice` subset contains 278 tasks: retail 114, airline 50, telecom 114.
`banking_knowledge` adds the `τ-Knowledge` tasks and uses the configured retrieval
backend. The `mock` domain remains excluded because it is primarily for tau2 testing.

## AWS Bedrock Setup

Prereqs:

- `uv`
- `git`
- Python available on PATH
- AWS credentials with Bedrock runtime access
- Bedrock model access enabled for Anthropic Claude Sonnet in your AWS account/region

Set credentials using your preferred AWS method. Examples:

```powershell
$env:AWS_REGION_NAME="us-east-1"
$env:AWS_DEFAULT_REGION="us-east-1"
$env:AWS_PROFILE="your-profile"
```

Or use explicit keys:

```powershell
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_SESSION_TOKEN="..."  # only if using temporary credentials
$env:AWS_REGION_NAME="us-east-1"
```

The default model string is:

```text
bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

If your Bedrock account uses a regional profile or different model ID, override it with
`--agent-model` and `--user-model`.

## Smoke Run

Start with one task in one cheap domain:

```powershell
python scripts\tau3_dfc_benchmark.py `
  --domains retail `
  --num-tasks 1 `
  --num-trials 1 `
  --max-concurrency 1 `
  --run-baseline
```

Then run the matching DFC condition and analyze:

```powershell
python scripts\tau3_dfc_benchmark.py `
  --domains retail `
  --num-tasks 1 `
  --num-trials 1 `
  --max-concurrency 1 `
  --skip-setup `
  --skip-generate `
  --run-dfc `
  --analyze
```

## Full Tau3 Run: One Trial, Every Task

This runs every task in each condition across `retail`, `airline`, `telecom`,
and `banking_knowledge`: first baseline, then DFC, followed by side-by-side utility
analysis. One trial and all tasks are script defaults. Banking uses `bm25` retrieval
unless `--retrieval-config` overrides it.

```powershell
python scripts\tau3_dfc_benchmark.py `
  --max-concurrency 3 `
  --full-run
```

In Colab:

```python
!python scripts/tau3_dfc_benchmark.py \
  --max-concurrency 3 \
  --full-run
```

Use `--max-concurrency 1` if Bedrock throttles requests.

## Full Baseline Only

```powershell
python scripts\tau3_dfc_benchmark.py `
  --max-concurrency 3 `
  --run-baseline
```

Baseline results land under:

```text
external/tau2-bench/data/simulations/baseline_bedrock_sonnet_<domain>/results.json
```

## Full DFC Run Only

Review this first:

```text
generated/tau3/generated_tau3_dfc_policies.json
```

Then:

```powershell
python scripts\tau3_dfc_benchmark.py `
  --max-concurrency 3 `
  --skip-setup `
  --skip-generate `
  --run-dfc `
  --analyze
```

DFC results land under:

```text
external/tau2-bench/data/simulations/dfc_bedrock_sonnet_<domain>/results.json
```

## Analysis Only

```powershell
python scripts\tau3_dfc_benchmark.py `
  --skip-setup `
  --skip-generate `
  --analyze
```

Open:

```text
generated/tau3/analysis/summary.csv
generated/tau3/analysis/by_task.csv
```

Key columns:

- `baseline_pass_rate`
- `dfc_pass_rate`
- `delta_pass_rate`
- `baseline_avg_reward`
- `dfc_avg_reward`
- `delta_avg_reward`

Positive delta means utility improved under DFC. Negative delta means the sink map or trusted-value
materializer is probably over-blocking and needs domain-specific tightening.

## Banking-Only Run

To run only the Sierra `τ-Knowledge` banking domain:

```powershell
python scripts\tau3_dfc_benchmark.py `
  --domains banking_knowledge `
  --retrieval-config bm25 `
  --num-trials 4 `
  --all-tasks `
  --max-concurrency 1 `
  --run-baseline
```

```powershell
python scripts\tau3_dfc_benchmark.py `
  --domains banking_knowledge `
  --retrieval-config bm25 `
  --num-trials 4 `
  --all-tasks `
  --max-concurrency 1 `
  --skip-setup `
  --skip-generate `
  --run-dfc `
  --analyze
```

## Useful Overrides

Use a different Bedrock model/profile:

```powershell
python scripts\tau3_dfc_benchmark.py `
  --agent-model "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0" `
  --user-model "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0" `
  --run-baseline
```

Use a different existing tau2 checkout:

```powershell
python scripts\tau3_dfc_benchmark.py --repo-dir C:\path\to\tau2-bench --run-baseline
```

Use only specific task IDs:

```powershell
python scripts\tau3_dfc_benchmark.py --domains airline --task-ids task_0 task_1 --run-baseline
```

Pass an extra tau2 CLI token:

```powershell
python scripts\tau3_dfc_benchmark.py --tau-extra-arg=--verbose-logs --run-baseline
```
