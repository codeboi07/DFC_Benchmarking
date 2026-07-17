# Colab quickstart: full Spider2-DBT Sales comparison

This is the exact flow I’d use in Colab for your small-vs-big experiment.

Models:

```text
big   = Claude Opus 4.8      anthropic.claude-opus-4-8
small = Qwen3 235B A22B 2507 qwen.qwen3-235b-a22b-2507-v1:0
```

## 0. Runtime

Use a normal Colab CPU runtime first. These tasks run local DuckDB/dbt; the LLM calls go to Bedrock.

## 1. Install dependencies

```python
!pip install -q boto3 duckdb pandas pyyaml dbt-core dbt-duckdb gdown
```

## 2. Clone repos

If your `DFC_Benchmarking` repo is on GitHub, clone it. If not, upload the `spider2_dbt_compare/` folder manually to `/content/DFC_Benchmarking`.

```python
%cd /content
!git clone https://github.com/xlang-ai/Spider2.git

# Replace this with your repo URL if it is pushed to GitHub.
# !git clone <YOUR_DFC_BENCHMARKING_REPO_URL> DFC_Benchmarking
```

If you manually uploaded this folder, set:

```python
%cd /content/DFC_Benchmarking
```

If cloned:

```python
%cd /content/DFC_Benchmarking
!pip install -q -r spider2_dbt_compare/requirements.txt
```

## 3. Download Spider2-DBT source databases

The official Spider2-DBT README says to download two Google Drive files from inside `spider2-dbt`:

```python
%cd /content/Spider2/spider2-dbt
!gdown 'https://drive.google.com/uc?id=1N3f7BSWC4foj-V-1C9n8M2XmgV7FOcqL'
!gdown 'https://drive.google.com/uc?id=1s0USV_iQLo4oe05QqAMnhGGp5jeejCzp'
!python setup.py
```

After setup, find the extracted task directory:

```python
!find /content/Spider2/spider2-dbt -maxdepth 3 -type d -name 'salesforce001' -print
```

You need the parent directory that contains task folders such as:

```text
salesforce001/
hubspot001/
scd001/
...
```

Common candidates:

```text
/content/Spider2/spider2-dbt/DBT_start_db
/content/Spider2/spider2-dbt/dbt_start_db
/content/Spider2/spider2-dbt
```

Set it:

```python
TASK_ROOT = "/content/Spider2/spider2-dbt/DBT_start_db"  # change if find shows another parent
MANIFEST = "/content/Spider2/spider2-dbt/examples/spider2-dbt.jsonl"
```

Sanity check:

```python
import os
for t in ["salesforce001", "hubspot001", "scd001"]:
    print(t, os.path.exists(os.path.join(TASK_ROOT, t)))
```

## 4. Configure AWS Bedrock credentials

Use whatever credential method you already have in Colab. For environment variables:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_SESSION_TOKEN"] = "..."  # only if using temporary creds
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
```

Quick model-access check:

```python
import boto3
client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

for model_id in [
    "anthropic.claude-opus-4-8",
    "qwen.qwen3-235b-a22b-2507-v1:0",
]:
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with OK only."}]}],
            inferenceConfig={"maxTokens": 16, "temperature": 0},
        )
        print(model_id, resp["output"]["message"]["content"][0]["text"])
    except Exception as e:
        print("FAILED", model_id, repr(e))
```

If Opus fails in `us-east-1`, try the global inference profile:

```text
global.anthropic.claude-opus-4-8
```

The runner lets you override:

```text
--big-model-id global.anthropic.claude-opus-4-8
```

## 5. Smoke test one task, one model

Run Qwen on `twilio001` first because it is small and cheap-ish:

```python
%cd /content/DFC_Benchmarking
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root "$TASK_ROOT" \
  --manifest "$MANIFEST" \
  --tasks twilio001 \
  --models small \
  --trials 1 \
  --max-steps 20 \
  --region "$AWS_REGION"
```

Check output:

```python
!find runs/spider2-dbt-sales/twilio001 -maxdepth 5 -type f | sort
```

Open the trace:

```python
import json, glob
trace_path = glob.glob("runs/spider2-dbt-sales/twilio001/small_qwen235b/trial_1/agent/*.json")[0]
trace = json.load(open(trace_path))
print(trace["instance_id"], trace["model"]["model_id"], trace["evaluation"])
print("steps:", len(trace["trajectory"]))
print(trace["trajectory"][0].keys())
```

## 6. Smoke test both models on one hard task

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root "$TASK_ROOT" \
  --manifest "$MANIFEST" \
  --tasks salesforce001 \
  --models small big \
  --trials 1 \
  --max-steps 30 \
  --region "$AWS_REGION"
```

## 7. Full Sales run

One trial for all 13 Sales tasks:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root "$TASK_ROOT" \
  --manifest "$MANIFEST" \
  --trials 1 \
  --max-steps 30 \
  --region "$AWS_REGION"
```

Research-quality run with 3 trials:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root "$TASK_ROOT" \
  --manifest "$MANIFEST" \
  --trials 3 \
  --max-steps 30 \
  --region "$AWS_REGION"
```

## 8. Rebuild comparison CSVs

If a run is interrupted, you can aggregate whatever finished:

```python
!python spider2_dbt_compare/run_sales_compare.py \
  --task-root "$TASK_ROOT" \
  --out-root runs/spider2-dbt-sales \
  --aggregate-only
```

The two main files:

```text
runs/spider2-dbt-sales/sales_comparison.csv
runs/spider2-dbt-sales/sales_comparison_paired.csv
```

Preview:

```python
import pandas as pd
df = pd.read_csv("runs/spider2-dbt-sales/sales_comparison.csv")
display(df)

paired = pd.read_csv("runs/spider2-dbt-sales/sales_comparison_paired.csv")
display(paired)
```

## 9. Zip/download traces

```python
!zip -qr spider2_dbt_sales_runs.zip runs/spider2-dbt-sales
from google.colab import files
files.download("spider2_dbt_sales_runs.zip")
```

## 10. What to analyze first

Start with these columns:

```text
instance_id
model
trial
finished
steps
dbt_run_count
dbt_run_success
file_edit_count
sql_query_count
inspected_yaml
inspected_models
error_type
```

Then manually inspect trajectories where:

```text
small_qwen235b failed but big_opus48 succeeded
small_qwen235b used more steps
small_qwen235b did no file edits
small_qwen235b ran dbt but did not verify output
```

