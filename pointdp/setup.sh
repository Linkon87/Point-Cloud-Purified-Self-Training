set -e
PYTHON_BIN=${PYTHON_BIN:-python}

abspath=$(readlink -f "$0")
root=$(dirname $abspath)
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"

cd ${root}/pointnet2_pyt && ${PYTHON_BIN} -m pip install -e . && cd -
cd ${root}/PCT_Pytorch/pointnet2_ops_lib && ${PYTHON_BIN} setup.py install && cd -
cd ${root}/emd && ${PYTHON_BIN} setup.py install && cd -
