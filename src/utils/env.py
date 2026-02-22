"""Environment variable loading helpers."""

from __future__ import annotations

from dotenv import load_dotenv


def load_env_file() -> None:
    """Load local .env without overriding existing process variables."""

    load_dotenv(override=False)
