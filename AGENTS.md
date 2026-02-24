# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains runtime code, organized by domain: `algos/`, `models/`, `data/`, `runners/`, `logging/`, `core/`, and `utils/`.
- `scripts/` contains CLI entry points: `train.py` and `eval.py`.
- `configs/` stores Hydra configuration groups (`algo/`, `data/`, `model/`, `logger/`) plus root `config.yaml`.
- `tests/` is split into `tests/unit/` and `tests/integration/`.
- `data/` is a local symlink to processed datasets; avoid committing large datasets, logs, or checkpoints.

## Build, Test, and Development Commands
- `source .venv/bin/activate`: activate the project virtual environment before running python related command.
- `python scripts/train.py`: run default offline training.
- `python scripts/train.py algo=bc_il model=mlp_actor`: run training with Hydra overrides.
- `python scripts/eval.py checkpoint_path=/abs/path/policy.pth`: evaluate a saved policy checkpoint.
- `python -m pytest -q`: run all tests.
- `python -m pytest -m integration -q`: run end-to-end smoke tests only.
- `ruff check .`: lint Python code.

## Coding Style & Naming Conventions
- Target Python `>=3.12`; use 4-space indentation and follow PEP 8.
- Prefer explicit type hints on public functions and module boundaries.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep new components aligned with registry patterns (`src/algos/registry.py`, `src/models/registry.py`).
- Use lowercase snake_case for config filenames (example: `configs/algo/td3_bc.yaml`).

## Testing Guidelines
- Framework: `pytest`; integration scenarios are marked with `@pytest.mark.integration`.
- File names should follow `test_<feature>.py`; test functions should state behavior (`test_train_then_eval_checkpoint`).
- Add or update unit tests for local logic changes and integration smoke tests for pipeline-level changes.
- No fixed coverage threshold is enforced yet; prioritize regression coverage for data adapters, schema mapping, and train/eval flows.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (consistent with existing history).
- Keep commits scoped and include related tests/config updates in the same change.
- PRs should include: intent, key commands used to validate, and relevant test output summary.
- Link issue/task IDs when available and call out config or data assumptions explicitly.

## Security & Configuration Tips
- Use `.env` for local configuration; do not commit secrets.
- Common environment variables: `SPRL_DATA_PATH`, `SPRL_LOGDIR`, `SPRL_WANDB_PROJECT`, `SPRL_CHECKPOINT_PATH`.
- Prefer absolute paths for checkpoint evaluation to avoid path confusion under Hydra run directories.
