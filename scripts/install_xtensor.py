import subprocess
import sys
import pathlib
import pybind11
import sysconfig

lib_name = sys.argv[1]
tag_name = sys.argv[2]

pybind11.get_cmake_dir()

repo_url = 'https://github.com/xtensor-stack/' + lib_name

home = pathlib.Path.home()
paths = sysconfig.get_paths()

subprocess.check_call(['git', 'clone', repo_url], cwd=home)
subprocess.check_call(['git', 'checkout', f'tags/{tag_name}', '-b', tag_name], cwd=home / lib_name)
build = home / lib_name / "build"
build.mkdir()
subprocess.check_call([
	'cmake',
	f'-DPYTHON_INCLUDE_DIR={paths["include"]}',
	f'-DPYTHON_EXECUTABLE={sys.executable}',
	'..'], cwd=build)
subprocess.check_call(['make', 'install'], cwd=build)
