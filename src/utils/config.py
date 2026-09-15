from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(path: Path | str) -> DictConfig:
    """Loads YAML and converts it into YAML hierachy."""
    if isinstance(path, str):
        path = Path(path)

    return OmegaConf.load(path)
