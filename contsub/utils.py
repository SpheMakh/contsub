import numpy as np
from casacore.measures import measures
from typing import Dict
import click

def direction_rad(ra, dec, epoch="J2000"):
    dm = measures()
    radec = []
    for val in [ra, dec]:
        if isinstance(val, str):
            try:
                new_val = float(val)
                new_val += "deg"
            except ValueError:
                new_val = val
        else:
            new_val = f"{val}deg"
        
        radec.append(new_val)
    
    new_radec = dm.direction(epoch, radec[0], radec[1])
    return new_radec["m0"]["value"], new_radec["m1"]["value"] 

    
def radec2lm(ra0: float, dec0: float,
            ra: float, dec: float):
    """ Convert RA, Dec coordinate to lm-plane coordinates. Inputs should be in radians

    Args:
        ra0 (float): Field centre RA.
        dec0 (float): Field centre Dec
        ra (float): RA
        dec (float): Dec

    Returns:
        tuple: lm-plane direction tuple in radians
    """
    
    dra = ra - ra0
    skyl = np.cos(dec) * np.sin(dra) 
    skym = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    
    return skyl, skym

def phase_shift(vis, uvw, l0, m0):
    
    n0 = np.sqrt(1 - l0*l0 - m0*m0)
    phase = uvw[0] * l0 + uvw[1] * m0 + uvw[2] * n0
    
    return vis * np.exp(-2j * np.pi * phase)


def parse_dim_dict(dims: str) -> Dict[str, int]:
        msg = f"'{dims}' does not conform to {{d1: s1, d2: s2, ..., dn: sn}}"
        err = click.BadOptionUsage("chunks", msg)
        dims = dims.strip()

        if not dims[0] == "{" and dims[-1] == "}":
            raise err

        result = {}

        for dim in [d.strip() for d in dims[1:-1].split(",")]:
            bits = [b.strip() for b in dim.split(":")]

            if len(bits) != 2:
                raise err

            try:
                result[bits[0]] = int(bits[1])
            except ValueError:
                raise err

        return result