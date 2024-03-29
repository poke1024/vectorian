yum install -y git cmake
# specifing a wrong numpy version here will end up in runtime errors like:
# module compiled against API version 0xe but this version of numpy is 0xd

python -m pip install numpy~=1.19
python -m pip install "pybind11~=2.6.2"

python -m pip install "numpy[global]~=1.19"
python -m pip install "pybind11[global]~=2.6.2"

python scripts/install_xtensor.py xtl 0.7.2
python scripts/install_xtensor.py xtensor 0.23.4
python scripts/install_xtensor.py xsimd 7.4.9
python scripts/install_xtensor.py xtensor-python 0.25.1
