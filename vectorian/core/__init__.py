import os

__pdoc__ = {
	'vectorian_core': False,
	'cpp': False,
	'lib': False
}


def build_vectorian():
	import cppimport

	cppimport.set_quiet(False)
	cppimport.force_rebuild(False)

	if os.environ.get("DEBUG_VECTORIAN_FORCE_REBUILD", False):
		cppimport.force_rebuild(True)


if os.environ.get("VECTORIAN_CPP_IMPORT", False):
	import logging
	logging.debug("cpp core: importing...")

	build_vectorian()

	import cppimport.import_hook
	from .cpp.core import *

	logging.debug("cpp core: imported.")
else:
	from vectorian_core import *
