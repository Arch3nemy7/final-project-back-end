"""Database layer (SQLAlchemy 2.0).

Stores lightweight *run metadata* only — name, dataset, the StyleGAN config, the
current stage/status. The heavy artifacts (snapshots, sample images, metric
logs) stay on disk in the run directory, which is the real source of truth. That
is why a small SQLite file is plenty; you do not need PostgreSQL for this.
"""
import json
from datetime import datetime, timezone

from sqlalchemy import String, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="Untitled run")
    dataset: Mapped[str] = mapped_column(String(300), default="—")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    # Full front-end pipeline state (stage/done/results/…). Lets the server hold
    # a complete run — e.g. one imported from already-trained models — so the UI
    # can adopt it.
    pipe_json: Mapped[str] = mapped_column(Text, default="{}")
    stage: Mapped[str] = mapped_column(String(40), default="format")
    status: Mapped[str] = mapped_column(String(40), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "dataset": self.dataset,
            "config": json.loads(self.config_json or "{}"),
            "pipe": json.loads(self.pipe_json or "{}"),
            "stage": self.stage,
            "status": self.status,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }


# SQLite needs check_same_thread=False because FastAPI may touch the session
# from worker threads; harmless for other backends.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def upsert_run(session, run_id: str, *, name=None, dataset=None, config=None, pipe=None) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        run = Run(id=run_id)
        session.add(run)
    if name is not None:
        run.name = name
    if dataset is not None:
        run.dataset = dataset
    if config is not None:
        run.config_json = json.dumps(config)
    if pipe is not None:
        run.pipe_json = json.dumps(pipe)
    session.commit()
    return run
