# tracking/

Logs training and evaluation to **Weights & Biases** and/or **MLflow**. Nothing is logged until keys are set.

```bash
cp .env.example .env     # then fill in the keys; .env is git-ignored
```

Training and evaluation read `.env` at start. A variable already set in the shell wins.

| Tracker | On when | Where runs go |
|---|---|---|
| W&B | `WANDB_API_KEY` is set | project `WANDB_PROJECT`, team `WANDB_ENTITY` |
| MLflow | `MLFLOW_TRACKING_URI` is set | experiment `MLFLOW_EXPERIMENT_NAME` (login: `MLFLOW_TRACKING_USERNAME` / `PASSWORD`, or `MLFLOW_TRACKING_TOKEN`) |

## What is logged

| Run | Name | Content |
|---|---|---|
| Training | the config's `name`, e.g. `stage1a_printed` | loss, learning rate, grad norm, dev loss (every `logging_steps` / `eval_steps`), training arguments. Each saved checkpoint is uploaded if `WANDB_LOG_MODEL=checkpoint` / `HF_MLFLOW_LOG_ARTIFACTS=TRUE` |
| Evaluation | `eval <config> <split> <checkpoint>`, one per `--checkpoint` | every score group (e.g. `source.mozhi_hi/cer`, `script.handwritten/lang.hi/cer`), gate results (`gate/...` = 1 pass / 0 fail, `gates_failed`), params (checkpoint, model path, before/after), `report_<split>.json` |

**Linking runs:**
- **W&B:** runs of one config share a group, with job type `train` or `eval`.
- **MLflow:** runs carry the tags `config`, `stage` and `kind` (`train` / `eval`). Filter in the UI with, for example, `tags.stage = "1A"`.

**Not logged:**
- Predictions and `report_<split>.md`, because they quote private text.
- `--smoke` and `--profile` runs.
- `merged/` models (6.7 GB each).

**Privacy:** checkpoints from Stage 1C on are trained on golden data (personal information). Upload them only to a server your team controls, or set the two upload variables to `false` / `FALSE`.

## Shared MLflow server for the team (GPU server, port 9898)

The team opens it in a browser and logs in, with no SSH tunnel. Run these on the GPU server, inside the repo's venv: `requirements.txt` installs MLflow 3.x, server included.

**1. Login config, once.** Copy MLflow's default and change two lines:

```bash
source ~/ocrTraining/.venv/bin/activate
mkdir -p ~/mlflow
cp "$(python -c 'import mlflow.server.auth as a, pathlib; print(pathlib.Path(a.__file__).parent / "basic_auth.ini")')" ~/mlflow/auth.ini
chmod 600 ~/mlflow/auth.ini
nano ~/mlflow/auth.ini
#   admin_password = <strong password>                            (uncomment: MLflow 3 has no default and won't start without one)
#   database_uri   = sqlite:////home/<you>/mlflow/basic_auth.db   (absolute path: 4 slashes)
```

**2. Start the server** in tmux so it survives logout:

```bash
tmux new -s mlflow
source ~/ocrTraining/.venv/bin/activate
MLFLOW_AUTH_CONFIG_PATH=$HOME/mlflow/auth.ini mlflow server --app-name basic-auth \
    --host 0.0.0.0 --port 9898 \
    --backend-store-uri sqlite:///$HOME/mlflow/mlflow.db \
    --artifacts-destination $HOME/mlflow/artifacts
```

**3. Create users, once.** Make `trainer` for training and eval jobs, and one login per teammate:

```bash
MLFLOW_TRACKING_USERNAME=admin MLFLOW_TRACKING_PASSWORD='<admin password>' python - <<'EOF'
from mlflow.server import get_app_client
users = get_app_client("basic-auth", tracking_uri="http://127.0.0.1:9898")
users.create_user(username="trainer", password="<password>")
users.create_user(username="<teammate>", password="<password>")
EOF
```

New users can read every experiment (`default_permission = READ`): teammates can view runs but not delete them. `trainer` owns the experiments it creates.

**4. The GPU server's `.env`:**

```
MLFLOW_TRACKING_URI=http://127.0.0.1:9898
MLFLOW_TRACKING_USERNAME=trainer
MLFLOW_TRACKING_PASSWORD=<password>
MLFLOW_EXPERIMENT_NAME=hindi-doc-ocr
```

**5. Team access:** open `http://<server-ip>:9898` and log in.
- **By IP on the LAN:** works as is. MLflow 3 accepts private IPs (10.\*, 192.168.\*, 172.16–31.\*) and localhost.
- **By a hostname:** add `--allowed-hosts "<hostname>:9898,<server-ip>:9898"` to the server command.
- **Checkpoints:** the training run lists them under **Logged models** (`checkpoint-1000`, …), not under Artifacts.

If the page doesn't load, the server firewall is blocking the port. Open it for the team's network only, for example `sudo ufw allow from <team subnet> to any port 9898 proto tcp`, or ask the server admin.

**Security:**
- Anyone who can reach port 9898 sees the login page, so keep the port on the internal network or VPN.
- Plain HTTP sends passwords and data unencrypted. For access from outside, put HTTPS (e.g. nginx) in front.
- Every logged-in user can download uploaded checkpoints, including Stage 1C+ models trained on golden data.

Code: `__init__.py` (`load_env`, `enabled`, `log_eval`). Tests: `tests/test_tracking.py`.
