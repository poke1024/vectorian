import numpy as np
import yaml
import os

from distutils.core import setup
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('environment.yml') as f:
	required = yaml.load(f.read())['dependencies'][-1]['pip']

sources = [str(x) for x in Path('vectorian/core/cpp').rglob('*.cpp')]
sources += ['vectorian/core/lib/ppk_assert/src/ppk_assert.cpp']

include_dirs = [np.get_include()]
include_dirs += ['vectorian/core/lib/pyemd/pyemd/lib']
include_dirs += ['vectorian/core/lib/ppk_assert/src']
include_dirs += ['vectorian/core/cpp']

macros = [('VECTORIAN_SETUP_PY', '1')]
if os.environ.get("VECTORIAN_BLAS", False):
	macros += [('VECTORIAN_BLAS', '1')]

ext_modules = [
	Pybind11Extension(
		"vectorian_core",
		sorted(sources),
		cxx_std=17,
		define_macros=macros,
		include_dirs=include_dirs,
	),
]

setup(
	name='Vectorian',
	version='0.1dev',
	packages=['vectorian'],
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='poke1024@gmx.org',
	long_description='',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=required,
)