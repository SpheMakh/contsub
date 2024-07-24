import numpy as np
import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File, MS
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
from contsub.uvplane import ms_subtract

log = init_logger(BIN.uv_plane)

command = BIN.uv_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

@click.command(command)
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    ms = MS(opts.ms)
    
    fitopts = {
        "func": opts.fit_model,
        "order": opts.order[0],
        "segments": opts.segments[0],
        #"sigma_clip": opts.sigma_clip,
        "xvar": None,
    }
    
    ms_subtract(ms, incol=opts.incol, outcol=opts.outcol, field=opts.field,
                spwid=opts.spw, weight_column=opts.weight_column,
                fitopts=fitopts)
    
    
    