/*cppimport
<%
import pyarrow
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
	os.path.join(pyarrow.get_library_dirs()[0], 'include'),
	numpy.get_include()]

if platform.system() == 'Linux':
	cfg['libraries'] = ['arrow', 'arrow_python']
	cfg['linker_args'] = ['-L' + pyarrow.get_library_dirs()[0]]

cfg['include_dirs'].append('../../../lib')
#cfg['include_dirs'].append('../../../lib/memory/include')
#cfg['include_dirs'].append('../../../lib/memory/src')
cfg['include_dirs'].append('../../../lib/ppk_assert/src')

old_pyarrow = 1 if compare_versions(
	pyarrow.__version__, '0.12.1', operator.le) else 0

cfg['compiler_args'] = [
	'-Wall',
	'-std=c++17',
	'-DPYARROW_0_12_1=%d' % old_pyarrow]
cfg['extra_link_args'] = []

if platform.system() == 'Linux':
	cfg['compiler_args'].append('-std=c++1z')  # GNU-C C++17 support
	os.environ["CC"] = "gcc-8"
	os.environ["CXX"] = "gcc-8"
	cfg['compiler_args'].append('-D_GLIBCXX_USE_CXX11_ABI=1')

if os.environ.get('DEBUG_VECTORIAN_CORE', False):
	logging.info("compiling for debugging.")

	cfg['compiler_args'].append('-g')
	cfg['compiler_args'].append('-O1')

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
	'alignment/wmd.h',
	'alignment/wrd.h',
	'embedding/contextual.h',
	'embedding/embedding.h',
	'embedding/sim.h',
	'embedding/static.h',
	'embedding/vectors.h',
	'match/match.h',
	'match/match_impl.h',
	'match/matcher.h',
	'match/matcher_impl.h',
	'match/region.h',
	'metric/alignment.h',
	'metric/composite.h',
	'metric/contextual.h',
	'metric/factory.h',
	'metric/metric.h',
	'metric/static.h',
	'slice/static.h',
	'common.h',
	'document.h',
	'query.h',
	'result_set.h',
	'utils.h',
	'vocabulary.h'
]

cfg['sources'] = [
	'match/flow.cpp',
	'match/match.cpp',
	'match/matcher.cpp',
	'metric/static.cpp',
	'embedding/sim.cpp',
	'module.cpp',
	'embedding/static.cpp',
	'metric/contextual.cpp',
	'common.cpp',
	'result_set.cpp',
	'document.cpp',
	'query.cpp',
	'vocabulary.cpp',
	'../../../lib/ppk_assert/src/ppk_assert.cpp']

if platform.system() == 'Darwin':  # >= macOS 10.14.6
	cfg['compiler_args'].append("-stdlib=libc++")
	cfg['extra_link_args'].append("-stdlib=libc++")

	# see https://github.com/pybind/python_example/issues/44
	cfg['compiler_args'].append("-mmacosx-version-min=10.15")
	cfg['linker_args'].append("-mmacosx-version-min=10.15")

logging.debug(f"cpp core: compiling with {cfg}")

setup_pybind11(cfg)
%>
*/
