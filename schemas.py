"""Request/response models for the API."""
from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    id: str
    name: str = "Untitled run"
    dataset: str = "—"
    config: dict = Field(default_factory=dict)
    pipe: dict | None = None


class FormatStart(BaseModel):
    pos: str | None = None        # gram_positive.zip filename (staged in DATA_DIR)
    neg: str | None = None        # gram_negative.zip filename
    res: int = 256


class TrainStart(BaseModel):
    config: dict = Field(default_factory=dict)


class GenerateStart(BaseModel):
    n: int = 5000


class FidelityStart(BaseModel):
    """Fidelity = per-checkpoint FID sweep against the real crops."""
    num: int = 10000            # sweep samples per checkpoint; best is re-scored at fid50k_full
    gn: str | None = None       # gram-negative real dataset (zip/folder) at the model resolution
    gp: str | None = None       # gram-positive real dataset


class FeasibilityStart(BaseModel):
    """Feasibility = 5-CNN x 4-scenario macro-F1 study against real datasets.
    Synthetic crops come from this run's Generate output; the two real splits
    are configurable (fall back to the server's DATA_DIR/real and /test)."""
    real: str | None = None     # real TRAINING crops dir (gram_negative/ + gram_positive/)
    test: str | None = None     # isolated real TEST split dir (same two-subfolder layout)


class ImportModels(BaseModel):
    gn: str                 # gram-negative trained run directory (absolute path)
    gp: str                 # gram-positive trained run directory
    name: str = "Imported · trained generators"
    id: str | None = None   # optional run id; auto-generated if omitted
