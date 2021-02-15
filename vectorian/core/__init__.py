import logging
import os
import cppimport
import pyarrow

logging.debug("cpp core: importing...")

cppimport.set_quiet(False)
cppimport.force_rebuild(False)

# the following line is important on macOS so that the cppimport does not fail with a dyld error.
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ";" + ";".join(pyarrow.get_library_dirs())

import cppimport.import_hook
from .cpp.core import *

init_pyarrow()

logging.debug("cpp core: imported.")
