# https://docs.conda.io/projects/conda/en/latest/user-guide/install/rpm-debian.html
rpm --import https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc

cat <<EOF > /etc/yum/repos.d/conda.repo
[conda]
name=Conda
baseurl=https://repo.anaconda.com/pkgs/misc/rpmrepo/conda
enabled=1
gpgcheck=1
gpgkey=https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc
EOF

yum install -y conda

# Useful for debugging any issues with conda
conda info -a

conda install -c conda-forge xtl xtensor xsimd xtensor-blas xtensor-python
