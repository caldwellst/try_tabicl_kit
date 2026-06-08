# Try TabICL

Tiny self-contained kit for trying [TabICL](https://github.com/soda-inria/tabicl) — a pretrained tabular foundation model — on your own dataset. No fine-tuning, no hyperparameter search, no class weighting.

## What's in here

- `try_tabicl.ipynb` — the notebook. One CONFIG cell at the top, then load → split → fit → score.
- `pyproject.toml` — pinned dependency set (TabICL + scikit-learn + pandas + pyarrow + s3fs + JupyterLab).
- `uv.lock` — exact versions used to build this kit.
- `.python-version` — Python 3.13.

## Setup

You need [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

That's it. uv creates a virtualenv, pulls TabICL and friends, and resolves Python 3.13 if you don't already have it.

## Run

```bash
uv run jupyter lab try_tabicl.ipynb
```

Then **Run All**. The notebook ships with `DATA = "sample"` which loads sklearn's breast-cancer dataset, so it works end-to-end with zero edits. First run downloads the TabICL checkpoint from Hugging Face (~100 MB) and caches it.

## Use your own data

Edit the CONFIG cell:

```python
DATA = "path/to/my_table.parquet"   # or .csv / .tsv / s3://bucket/key
TARGET = "the_column_to_predict"
```

Supports local `.csv`, `.tsv`, `.parquet` files and `s3://` URIs. Binary or multiclass classification.

## Notes

- **Train size cap**: TabICL feeds the whole train set in as context, so runtime is roughly linear in `MAX_TRAIN_ROWS`. Default cap is 10,000 — bump it if you have the patience.
- **CPU only**: works fine without a GPU. Add `device="cuda"` to the `TabICLClassifier(...)` call if you have one.
- **First run is slow**: model download + first inference compile. Subsequent runs are fast.
