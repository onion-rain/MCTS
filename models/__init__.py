from .cifar import *
from .tv import *
from .misc import *

from . import cifar
from . import tv
from . import misc

CIFAR_MODEL_NAMES = sorted(name for name in cifar.__dict__
                            if name.islower()
                            and not name.startswith("__")
                            and callable(cifar.__dict__[name]))
                            
TV_MODEL_NAMES = sorted(name for name in tv.__dict__
                            if name.islower()
                            and not name.startswith("__")
                            and callable(tv.__dict__[name]))
                            
MISC_MODEL_NAMES = sorted(name for name in misc.__dict__
                            if name.islower()
                            and not name.startswith("__")
                            and callable(misc.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(), 
                            set(CIFAR_MODEL_NAMES + TV_MODEL_NAMES + MISC_MODEL_NAMES)))
                            
                            