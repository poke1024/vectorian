yum install -y git cmake
pip install "pybind11[global]" "numpy[global]"
python scripts/install_xtensor.py xtl 0.7.2
python scripts/install_xtensor.py xtensor 0.23.4
python scripts/install_xtensor.py xsimd 7.4.9
python scripts/install_xtensor.py xtensor-python 0.25.1
