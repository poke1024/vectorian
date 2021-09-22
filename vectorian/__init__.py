import os

from vectorian._version import __version__

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

# check for h5py version to prevent producing wrong results.
# workaround for regression with h5py >= 3.0 (need more time
# for detailed analysis and eventual fixing). test cases are
# wrong ndcg computations in notebook under:
# https://github.com/poke1024/vectorian-notebook
import h5py

if h5py.__version__ != '2.10.0':
	# do not remove this check. you might experience bogus
	# results otherwise.
	raise RuntimeError(
		f"expected h5py version 2.10.0, got {h5py.__version__}")
