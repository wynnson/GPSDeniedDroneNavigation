from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(path: Path) -> DictConfig:
    """Loads YAML and converts it into YAML hierachy."""
    return OmegaConf.load(path)
