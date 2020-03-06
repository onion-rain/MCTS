# from .cifar import *
# from .tv import *
# from .misc import *
from .slimming import *

# from . import cifar
# from . import tv
# from . import misc
from . import slimming

# CIFAR_MODEL_NAMES = sorted(name for name in cifar.__dict__
#                             if name.islower()
#                             and not name.startswith("__")
#                             and callable(cifar.__dict__[name]))
                            
# TV_MODEL_NAMES = sorted(name for name in tv.__dict__
#                             if name.islower()
#                             and not name.startswith("__")
#                             and callable(tv.__dict__[name]))
                            
# MISC_MODEL_NAMES = sorted(name for name in misc.__dict__
#                             if name.islower()
#                             and not name.startswith("__")
#                             and callable(misc.__dict__[name]))
                            
SLIMMING_MODEL_NAMES = sorted(name for name in slimming.__dict__
                            if name.islower()
                            and not name.startswith("__")
                            and callable(slimming.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(), 
                            set(SLIMMING_MODEL_NAMES)))
                            
                            