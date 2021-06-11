import numpy as np
import yaml
import os

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('environment.yml') as f:
	required = yaml.safe_load(f.read())['dependencies'][-1]['pip']

base_path = Path('vectorian').resolve()
assert base_path.exists()

sources = list((base_path / 'core/cpp').rglob('*.cpp'))
sources += [base_path / 'core/lib/ppk_assert/src/ppk_assert.cpp']

include_dirs = [Path(np.get_include())]
include_dirs += [base_path / 'core/lib/pyemd/pyemd/lib']
include_dirs += [base_path / 'core/lib/ppk_assert/src']
include_dirs += [base_path / 'core/cpp']

macros = [('VECTORIAN_SETUP_PY', '1')]
if os.environ.get("VECTORIAN_BLAS", False):
	macros += [('VECTORIAN_BLAS', '1')]

ext_modules = [
	Pybind11Extension(
		"vectorian_core",
		[str(x) for x in sorted(sources)],
		cxx_std=17,
		define_macros=macros,
		include_dirs=[str(x) for x in include_dirs],
	),
]

setup(
	name='vectorian',
	version='0.8.4',
	packages=find_packages(),
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='liebl@informatik.uni-leipzig.de',
	long_description='',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=required,
)
