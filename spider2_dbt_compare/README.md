# Spider2-DBT Sales: Bedrock small-vs-big trajectory pipeline

This folder contains a Colab-friendly runner for comparing:

- **big**: Claude Opus 4.8 on Amazon Bedrock
- **small**: Qwen3 235B A22B 2507 on Amazon Bedrock

It focuses first on the 13 Spider2-DBT Sales / CRM / workforce tasks and saves Harbor-style full agent trajectories.

## Confirmed Bedrock model IDs

Use these defaults:

```text
big_opus48     anthropic.claude-opus-4-8
small_qwen235b qwen.qwen3-235b-a22b-2507-v1:0
```

Notes:

- AWS documents Claude Opus 4.8 as active, with Bedrock Runtime model ID `anthropic.claude-opus-4-8`.
- AWS documents Qwen3 235B A22B 2507 with Bedrock Runtime model ID `qwen.qwen3-235b-a22b-2507-v1:0`.
- Qwen 235B is not actually “small” in parameter count, but it is your smaller/open-ish comparison model relative to Opus for this study.

## Sales tasks

The runner defaults to:

```text
asana001
workday002
scd001
lever001
greenhouse001
qualtrics001
salesforce001
hubspot001
jira001
marketo001
workday001
twilio001
intercom001
```

## Colab setup

For the full copy-paste Colab flow, use [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md).

Minimal dependency install:

```python
!pip install -q boto3 duckdb pandas pyyaml dbt-core dbt-duckdb
```

Clone this repo / upload this folder, then install the runner requirements:

```python
%cd /content/DFC_Benchmarking
!pip install -q -r spider2_dbt_compare/requirements.txt
```

Configure AWS credentials in the way you already use in Colab. Example:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_SESSION_TOKEN"] = "..."  # if needed
os.environ["AWS_REGION"] = "us-east-1"
```

You also need the Spider2-DBT task directories extracted locally. The expected layout is:

```text
DBT_start_db/
  salesforce001/
    dbt_project.yml
    profiles.yml
    models/
    salesforce.duckdb
  hubspot001/
  ...
```

If you have the official `spider2-dbt.jsonl` manifest, pass it with `--manifest`; otherwise the runner still works but the task prompt will be less detailed.

## Run a smoke test

Start with one task and one trial:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root /content/DBT_start_db \
  --manifest /content/Spider2/spider2-dbt/examples/spider2-dbt.jsonl \
  --tasks salesforce001 \
  --trials 1 \
  --region us-east-1
```

Run all Sales tasks:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root /content/DBT_start_db \
  --manifest /content/Spider2/spider2-dbt/examples/spider2-dbt.jsonl \
  --trials 1 \
  --region us-east-1
```

Run only one model:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root /content/DBT_start_db \
  --tasks hubspot001 scd001 salesforce001 \
  --models small \
  --trials 1
```

## Outputs

Each run creates:

```text
runs/spider2-dbt-sales/
  salesforce001/
    small_qwen235b/
      trial_1/
        workspace/
        agent/
          spider-agent-dbt.trajectory.json
        logs/
          stdout.log
          stderr.log
          dbt_run.log
        artifacts/
          manifest.json
          run_results.json
        evaluation.json
    big_opus48/
      trial_1/
        ...
  sales_comparison.csv
  sales_comparison_paired.csv
```

The trajectory JSON contains:

- full task prompt
- full system/action-space prompt
- step-by-step observations
- model responses
- extracted thoughts
- parsed actions
- tool calls
- full tool results
- file changes
- final evaluation summary

## Important comparison columns

`sales_comparison.csv` gives one row per task/model/trial:

- `finished`
- `terminated`
- `steps`
- `dbt_run_count`
- `dbt_run_success`
- `file_edit_count`
- `sql_query_count`
- `inspected_yaml`
- `inspected_models`
- `error_type`

`sales_comparison_paired.csv` gives small-vs-big rows per task/trial.

## Recommended experimental settings

For a first clean analysis:

```text
temperature = 0
max_steps = 30
trials = 1 smoke, then 3 for real comparison
same task root, same prompt, same action space
```

After the smoke test passes, run:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root /content/DBT_start_db \
  --manifest /content/Spider2/spider2-dbt/examples/spider2-dbt.jsonl \
  --trials 3 \
  --max-steps 30 \
  --region us-east-1
```
