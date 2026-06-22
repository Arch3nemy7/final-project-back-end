"""Server configuration.

All values can be overridden with environment variables (or a `.env` file in
this directory). Field name == env var name, case-insensitive — e.g. the field
``database_url`` is read from ``DATABASE_URL``.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database -----------------------------------------------------------
    # Default is a local SQLite file (zero install). To use the PostgreSQL you
    # already have, set e.g.
    #   DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/gramsynth
    database_url: str = f"sqlite:///{(HERE / 'gramsynth.db').as_posix()}"

    # --- paths --------------------------------------------------------------
    # The NVlabs StyleGAN2-ADA checkout (holds train.py / generate.py / …).
    stylegan_dir: Path = HERE.parent / "stylegan2-ada-pytorch"
    # Per-run working directories (datasets, snapshots, samples) live here.
    runs_dir: Path = HERE / "runs"
    # Where the cropped class .zip archives are staged on the server.
    data_dir: Path = HERE / "data"
    # Python interpreter used to launch the training jobs (point this at the
    # env that has PyTorch + CUDA installed).
    python_bin: str = "python"

    # --- behaviour ----------------------------------------------------------
    # Comma-separated list of allowed browser origins (the front end).
    cors_origins: str = "http://localhost:5173,http://localhost:4173"
    # StyleGAN2-ADA reports 1 tick per this many kimg; used to scale ETA.
    kimg_per_tick: int = 4
    # How often (seconds) the telemetry tailer polls the run directory.
    poll_interval: float = 0.5
    # Set GS_MOCK=1 to launch scripts/mock_train.py instead of the real
    # train.py — lets you demo the full real-time pipeline without a GPU.
    mock: bool = False

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
