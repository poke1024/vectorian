/*cppimport
<%
import numpy
import os
import platform
import operator
import cppimport
import logging

def normalize_version(v):
    parts = [int(x) for x in v.split(".")]
    while parts[-1] == 0:
        parts.pop()
    return parts

def compare_versions(v1, v2, cmp):
    return cmp(normalize_version(v1), normalize_version(v2))

cfg['include_dirs'] = [
	numpy.get_include()]

cfg['include_dirs'].append('../lib')
cfg['include_dirs'].append('../lib/pyemd/pyemd/lib')
cfg['include_dirs'].append('../lib/ppk_assert/src')
cfg['include_dirs'].append('../lib/pyalign')

cfg['compiler_args'] = [
	'-Wall',
	'-std=c++17',
	'-ftemplate-backtrace-limit=0']
cfg['extra_link_args'] = []

if platform.system() == 'Linux':
	cfg['compiler_args'].append('-std=c++1z')  # GNU-C C++17 support
	os.environ["CC"] = "gcc-8"
	os.environ["CXX"] = "gcc-8"
	cfg['compiler_args'].append('-D_GLIBCXX_USE_CXX11_ABI=1')

if os.environ.get('DEBUG_VECTORIAN_CORE', False):
	logging.info("compiling for debugging.")

	cfg['compiler_args'].append('-g')
	cfg['compiler_args'].append('-O0')
	cfg['compiler_args'].append('-UNDEBUG')

	if os.environ.get('VECTORIAN_SANITIZE_ADDRESS', False):
		cfg['compiler_args'].append('-fsanitize=address')
		cfg['compiler_args'].append('-fno-omit-frame-pointer')
		cfg['compiler_args'].append('-fno-optimize-sibling-calls')
		cfg['extra_link_args'].append('-fsanitize=address')
else:
	cfg['compiler_args'].append('-O3')

# see https://github.com/pybind/pybind11/blob/master/docs/faq.rst
cfg['compiler_args'].append('-fvisibility=hidden')

cfg['dependencies'] = [
	'alignment/aligner.h',
	'alignment/bow.h',
	'alignment/transport.h',
	'alignment/wmd.h',
	'alignment/wrd.h',
	'embedding/contextual.h',
	'embedding/embedding.h',
	'embedding/static.h',
	'embedding/vectors.h',
	'match/match.h',
	'match/match_impl.h',
	'match/matcher.h',
	'match/matcher_impl.h',
	'match/region.h',
	'metric/alignment.h',
	'metric/modifier.h',
	'metric/contextual.h',
	'metric/factory.h',
	'metric/metric.h',
	'metric/static.h',
	'slice/static.h',
	'common.h',
	'document.h',
	'query.h',
	'result_set.h',
	'vocabulary.h'
]

cfg['sources'] = [
	'match/instantiate.cpp',
	'match/flow.cpp',
	'match/match.cpp',
	'query.cpp',
	'vocabulary.cpp',
	'metric/contextual.cpp',
	'metric/modifier.cpp',
	'metric/static.cpp',
	'match/matcher.cpp',
	'embedding/static.cpp',
	'module.cpp',
	'common.cpp',
	'result_set.cpp',
	'document.cpp',
	'../lib/ppk_assert/src/ppk_assert.cpp',
	'../lib/pyalign/pyalign/algorithm/factory.cpp']

if platform.system() == 'Darwin':  # >= macOS 10.14.6
	cfg['compiler_args'].append("-stdlib=libc++")
	cfg['extra_link_args'].append("-stdlib=libc++")

	# see https://github.com/pybind/python_example/issues/44
	cfg['compiler_args'].append("-mmacosx-version-min=10.15")
	cfg['linker_args'].append("-mmacosx-version-min=10.15")

#cfg['parallel'] = True

logging.debug(f"cpp core: compiling with {cfg}")

setup_pybind11(cfg)
%>
*/
