yum install -y conda
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda env create --file environment.yml
conda activate vectorian
