# CS230 Last Mile Routing - JSON Workflow Quickstart

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the official data under `cs230_data/`, e.g.:

- `cs230_data/almrrc2021-data-train/model_build_inputs/{route_data.json, package_data.json, actual_sequences.json}`
- `cs230_data/almrrc2021-data-eval/model_build_inputs/{route_data.json, package_data.json, actual_sequences.json}`

## Baseline (JSON)

```bash
python src/run_baseline_json.py --json_root cs230_data --split train
```

Outputs: `outputs/route_<RID>_baseline.csv`

## Learned Edges (JSON)

```bash
python src/run_learned.py \
  --json_root_train cs230_data --split_train train \
  --json_root_test cs230_data --split_test eval \
  --route_id <RID>
```

Outputs: `outputs/route_<RID>_learned.csv`

## Tests

```bash
pytest -q
```


