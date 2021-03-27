yum install -y conda

conda create --name vectorian
conda activate vectorian

# Useful for debugging any issues with conda
conda info -a

conda install -c conda-forge xtl xtensor xsimd xtensor-blas xtensor-python
