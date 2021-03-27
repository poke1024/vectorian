yum install -y conda

conda create --name vectorian
conda activate vectorian
conda info -a

conda install -y -c conda-forge xtl xtensor xsimd xtensor-blas xtensor-python
