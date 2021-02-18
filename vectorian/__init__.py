import os


def compile_for_debugging():
	os.environ['DEBUG_VECTORIAN_CORE'] = "1"
	import vectorian.core
