import os


def compile():
	import vectorian.core


def compile_for_debugging():
	os.environ['DEBUG_VECTORIAN_CORE'] = "1"
	import vectorian.core
