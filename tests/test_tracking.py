import json
import os

import pytest

import tracking
from tracking import load_env  # the real one: conftest swaps tracking.load_env for a no-op

TEST_VARS = ["MLFLOW_EXPERIMENT_NAME", "FROM_ENV", "QUOTED"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for variable in TEST_VARS:
        monkeypatch.delenv(variable, raising=False)


def test_load_env_reads_values_and_keeps_the_shell(repo_tmp, monkeypatch):
    env = repo_tmp / ".env"
    env.write_text(
        "# comment\n\nWANDB_API_KEY=abc=123\nexport QUOTED='a b'\nFROM_ENV=file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FROM_ENV", "shell")
    load_env(env)
    assert os.environ["WANDB_API_KEY"] == "abc=123"
    assert os.environ["QUOTED"] == "a b"
    assert os.environ["FROM_ENV"] == "shell"
    assert tracking.enabled() == ["wandb"]


def test_load_env_without_file(repo_tmp):
    load_env(repo_tmp / "missing.env")
    assert tracking.enabled() == []


def test_flatten_keeps_numbers_with_valid_names():
    scores = {
        "script=handwritten|lang=hi": {"cer": 0.5, "n": 3, "flag": True, "note": "x", "b": None},
        "source=pages": {"ap50": {"text": 0.25}},
    }
    assert tracking.flatten(scores) == {
        "script.handwritten/lang.hi/cer": 0.5,
        "script.handwritten/lang.hi/n": 3.0,
        "source.pages/ap50/text": 0.25,
    }


def test_gate_metrics():
    gates = [
        {"group": "source=a", "metric": "cer", "status": "pass"},
        {"group": "lang=hi", "metric": "cer", "status": "fail: rose 0.02"},
        {"group": "lang=en", "metric": "cer", "status": "skipped: group not evaluated"},
    ]
    assert tracking.gate_metrics(gates) == {
        "gate/source.a/cer": 1.0,
        "gate/lang.hi/cer": 0.0,
        "gates_failed": 1.0,
    }


def _eval_inputs(repo_tmp):
    raw = {"name": "stage1a_printed", "stage": {"id": "1A"}}
    report = repo_tmp / "report_dev.json"
    report.write_text(json.dumps({"scores": []}), encoding="utf-8")
    scores = [{"source=a": {"cer": 0.4, "n": 2}}, {"source=a": {"cer": 0.1, "n": 2}}]
    gates = [[{"group": "source=a", "metric": "cer", "status": "pass"}]]
    return raw, ["start", "runs/checkpoint-2"], ["base", "runs/checkpoint-2"], scores, gates, report


def test_log_eval_without_trackers_does_nothing(repo_tmp):
    raw, labels, paths, scores, gates, report = _eval_inputs(repo_tmp)
    tracking.log_eval(raw, "dev", labels, paths, scores, gates, report)  # no import, no error


def test_log_eval_to_mlflow(repo_tmp, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    # A folder store is enough to check the calls (MLflow 3 needs the opt-in; servers use SQLite)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (repo_tmp / "mlruns").resolve().as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    raw, labels, paths, scores, gates, report = _eval_inputs(repo_tmp)
    tracking.log_eval(raw, "dev", labels, paths, scores, gates, report)

    runs = mlflow.search_runs(experiment_names=["test"], order_by=["params.checkpoint"])
    assert list(runs["params.checkpoint"]) == ["runs/checkpoint-2", "start"]
    assert list(runs["params.role"]) == ["after", "before"]
    assert list(runs["metrics.source.a/cer"]) == [0.1, 0.4]
    assert list(runs["metrics.gate/source.a/cer"].fillna(-1)) == [1.0, -1]
    assert set(runs["tags.kind"]) == {"eval"} and set(runs["tags.stage"]) == {"1A"}
    artifacts = mlflow.artifacts.list_artifacts(run_id=runs["run_id"][0])
    assert [a.path for a in artifacts] == ["report_dev.json"]
