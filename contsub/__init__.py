from importlib import metadata
from omegaconf import OmegaConf

__version__ = metadata.version(__package__)

BIN = OmegaConf.create({
    "im_plane": "imcontsub",
    "uv_plane": "uvcontsub",
    "main": "contsub",
})