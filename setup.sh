#!/usr/bin/env bash
# Template command, from the workspace root: bash ContourCraft/setup.sh
# Dedicated RTX 5080 runtime; preserves the authors' Python 3.10/PyG stack.
# CUDA 13 and matching PyTorch/Warp/RAPIDS target this RTX 5080 runtime.
set -eo pipefail
CCRAFT_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CCRAFT_BUILD="$CCRAFT_REPO/../data/build/ccraft"
eval "$(conda shell.bash hook)"
conda create -n ccraft python=3.10 pip -y
conda activate ccraft
conda install -c conda-forge -y \
  cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 \
  libcublas-dev=13.1 libcusparse-dev=12.6 libcusolver-dev=12.0 \
  gcc_linux-64=13 gxx_linux-64=13 ninja cmake
conda deactivate
conda activate ccraft
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=12.0
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CUB_HOME="$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export NVCC_CCBIN="$CXX"
export MAX_JOBS=6
export FORCE_CUDA=1
conda env config vars set -n ccraft CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0
python -m pip install setuptools==75.8.0 wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-build-isolation -r "$CCRAFT_REPO/requirements.txt" --extra-index-url https://pypi.nvidia.com
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  --only-binary=:all: --no-index -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
python -m pip install --no-deps \
  'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0+cu13-py3-none-manylinux_2_28_x86_64.whl'
mkdir -p "$CCRAFT_BUILD"
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "$CCRAFT_BUILD/pytorch3d"
python -m pip install --no-build-isolation --no-deps "$CCRAFT_BUILD/pytorch3d"
git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git "$CCRAFT_BUILD/cuda-samples"
git clone --depth 1 https://github.com/Dolorousrtur/CCCollisions.git "$CCRAFT_BUILD/CCCollisions"
export CUDA_SAMPLES_INC="$CCRAFT_BUILD/cuda-samples/Common"
# Update removed PyTorch dispatch APIs only; preserve the authors' collision kernels.
sed -i 's/\.type(),/\.scalar_type(),/g' "$CCRAFT_BUILD/CCCollisions/src/"*.cu
sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g' "$CCRAFT_BUILD/CCCollisions/src/cccollisions.cpp"
# CCCL 3 removed the empty unary_function base; keep each functor's operator().
sed -Ei 's/ : public thrust::unary_function<[^>]*>//g' \
  "$CCRAFT_BUILD/CCCollisions/src/utils.cu" "$CCRAFT_BUILD/CCCollisions/src/collisions_continuous.cu"
python -m pip install --no-build-isolation --no-deps "$CCRAFT_BUILD/CCCollisions"
python -m pip check
