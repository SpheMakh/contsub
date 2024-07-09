import os
from scabha import configuratt
from scabha.cargo import Parameter, EmptyDictDefault
from omegaconf import OmegaConf
from typing import Dict, Any, List
from dataclasses import dataclass

thisdir = os.path.dirname(__file__)

class File(str):
    def __init__(self, path, check=True):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.exists = os.path.exists(path)
        self.dirname = os.path.dirname(path)
        self.basename = os.path.splitext(self.name)[0]

        if check:
            if not self.exists:
                raise FileNotFoundError(f"File {self.path} does not exist.")
            self.isfile = os.path.isfile(self.path)

    
class Directory(File):
    @property
    def isdir(self):
        if self.check and not os.path.isdir(self.path):
            raise FileNotFoundError(f"File {self.path} is not a directory. (does it exist?)")
        else:
            return True


@dataclass
class SchemaSpec:
    inputs: Dict[str,Parameter]
    outputs: Dict[str,Parameter]
    libs: Dict[str, Any] = EmptyDictDefault()

def load(parser: File, use_sources: List = []) -> Dict:
    """Load a scabha-style parameter defintion using.

    Args:
        name (str): Name of parameter definition to load
        use_sources (List, optional): Parameter definition dependencies 
        (a.k.a files specified via_include)

    Returns:
        Dict: Schema object
    """
    
    args_defn = OmegaConf.structured(SchemaSpec)
    struct_args, _ = configuratt.load_nested([parser.path], structured=args_defn,
                                             use_sources=use_sources, use_cache=False)
    schema = OmegaConf.create(struct_args)
    
    return schema[parser.basename]

def load_sources(sources: List[str|File]):
    __sources = [None]*len(sources)
    for i, src in enumerate(sources):
        if isinstance(src, str):
            if src.endswith((".yaml", ".yml")):
                sources[i] = File(src)
            else:
                try:
                    sources[i] = File(os.path.join(thisdir,f"{src}.yaml"))
                except FileNotFoundError:
                    raise FileNotFoundError(f"Name {src} does not match a known parameter file.")
                
        __sources[i], _ = configuratt.load(sources[i], use_cache=False)
    return __sources[i]

