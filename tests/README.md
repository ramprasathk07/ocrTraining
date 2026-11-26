# tests/

pytest suite. It runs on CPU with no model download: model-facing code is tested with small stubs or on the meta device.

```bash
pytest -q                      # everything
pytest tests/test_stages.py    # one file
ruff check . && ruff format --check . && pytest   # what must pass before a commit
```

| File | Covers |
|---|---|
| `conftest.py` | `GOLDEN` (test golden data), station `OVERRIDES` from `configs/data/test_data.toml`, `repo_tmp` fixture (temp folder inside the repo; the sandbox blocks the system temp) |
| `test_golden.py` | golden page conversion, validation, station parsing |
| `test_splits.py` | station-level split, locking |
| `test_sources.py` | converters (`golden_regions`, `labelled_folder`) |
| `test_text.py` | NFC, UTF-8 checks |
| `test_document_ir.py` | grounded block serialize / parse |
| `test_inputs.py` | OCR-2 token layout, image modes, label masking |
| `test_config.py` | run configs load; stage chain 1A → 1B → 1C |
| `test_stages.py` | stage tree and its error checks, `read_split`, training prompt registration |
| `test_sampling.py` | exact split counts; sample plan follows shares and curriculum, is deterministic, spreads repeats evenly |
| `test_training_parts.py` | vision-gradient fix, per-group learning rates, unmatched group refused, inventory components |
| `test_metrics.py` | CER / WER |
| `test_page_metrics.py` | IoU, AP, matching, reading order, page scoring |
| `test_report.py` | score grouping, gates, report |
| `test_prompts.py` | eval prompt registry, leaderboard, prompt mismatch warnings |
| `test_evaluation_runner.py` | end-to-end evaluation with a stub model |
| `test_profiling.py` | `--profile`: labelled ranges, recorded steps, trace files |
| `test_tracking.py` | `.env` loading, tracker switch, metric names; eval runs logged to a local MLflow store |

Tests marked `needs_golden` read `dataEngine/data/test_data/src_golden`. That data is git-ignored, so on a fresh clone without it they are skipped, not failed.
