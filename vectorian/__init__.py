import os


def compile(for_debugging=False, force_rebuild=False):
	if for_debugging:
		os.environ['DEBUG_VECTORIAN_CORE'] = '1'
	if force_rebuild:
		os.environ['DEBUG_VECTORIAN_FORCE_REBUILD'] = '1'
	import vectorian.core


import vectorian.embeddings as embeddings
import vectorian.importers as importers
import vectorian.session as session
import vectorian.metrics as metrics
import vectorian.alignment as alignment

# "metrics" is the older name, "similarity" the newer name
similarity = metrics
