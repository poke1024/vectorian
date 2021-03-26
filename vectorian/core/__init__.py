import logging
import os
import cppimport

logging.debug("cpp core: importing...")

cppimport.set_quiet(False)
cppimport.force_rebuild(False)

if os.environ.get("DEBUG_VECTORIAN_FORCE_REBUILD", False):
	cppimport.force_rebuild(True)

import cppimport.import_hook
from .cpp.core import *

logging.debug("cpp core: imported.")
