import numpy as np
import yaml
import os
import platform

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('environment.yml') as f:
	required = yaml.safe_load(f.read())['dependencies'][-1]['pip']

base_path = Path('vectorian').resolve()
assert base_path.exists()

sources = list((base_path / 'core/cpp').rglob('*.cpp'))
sources += [base_path / 'core/lib/ppk_assert/src/ppk_assert.cpp']

include_dirs = [
	Path(np.get_include()),
	base_path / 'core/lib/pyemd/pyemd/lib',
	base_path / 'core/lib/ppk_assert/src',
	base_path / 'core/lib/pyalign',
	base_path / 'core/cpp'
]

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix is not None:
	include_dirs.append(f"{conda_prefix}/include")

macros = [('VECTORIAN_SETUP_PY', '1')]
if os.environ.get("VECTORIAN_BLAS", False):
	macros += [('VECTORIAN_BLAS', '1')]

extra_compile_args = []
extra_link_args = []

if os.environ.get("VECTORIAN_DEBUG", False) or \
	os.environ.get('VECTORIAN_SANITIZE_ADDRESS', False):

	extra_compile_args.extend(['-O0', '-g', '-UNDEBUG'])

	if os.environ.get('VECTORIAN_SANITIZE_ADDRESS', False):
		extra_compile_args.extend([
			'-fsanitize=address',
			'-fno-omit-frame-pointer',
			'-fno-optimize-sibling-calls'
		])
		extra_link_args.append('-fsanitize=address')

		if platform.system() == 'Linux':
			extra_link_args.append('-static-libasan')
else:
	extra_compile_args.extend(['-O3'])

ext_modules = [
	Pybind11Extension(
		"vectorian_core",
		[str(x) for x in sorted(sources)],
		cxx_std=17,
		extra_compile_args=extra_compile_args,
		extra_link_args=extra_link_args,
		define_macros=macros,
		include_dirs=[str(x) for x in include_dirs],
	),
]

exec(open("vectorian/_version.py").read())

setup(
	name='vectorian',
	version=__version__,
	packages=find_packages(),
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='liebl@informatik.uni-leipzig.de',
	long_description='',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=required,
)
