import os


def compile(for_debugging=False, force_rebuild=False):
	if for_debugging:
		os.environ['DEBUG_VECTORIAN_CORE'] = '1'
	if force_rebuild:
		os.environ['DEBUG_VECTORIAN_FORCE_REBUILD'] = '1'
	import vectorian.core
