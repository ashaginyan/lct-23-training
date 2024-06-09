from omegaconf import OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    project_name: str
    experiment_name: str
    config: str
    model: str
    n_epochs: int

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
