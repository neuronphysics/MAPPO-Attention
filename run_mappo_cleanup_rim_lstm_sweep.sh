#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=100G  
#SBATCH --time=4-23:59:00
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/projects/def-irina/memole/logs/mappo-attention-cleanup-sweep-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-irina/memole/logs/mappo-attention-cleanup-sweep-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc python/3.10 mujoco cuda/12.2
module load mpi4py/3.1.6
module spider pyarrow/17.0.0
module load arrow
module load opencv/4.10.0
module load scipy-stack
module spider mkl
module spider rust/1.65.0
module load cmake
module load intel
DIR=/home/memole/projects/def-irina/memole/Lora

#virtualenv --no-download --clear /home/memole/meltingpot
source /home/memole/meltingpot/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --upgrade pip setuptools wheel
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index torch torchvision torchtext torchaudio
#pip install --no-index --no-cache-dir wandb
#pip install --no-cache-dir -r ~/projects/def-irina/memole/Lora/requirements.txt
#pip install --no-index --no-cache-dir peft
#python -m pip install git+https://github.com/mpi4py/mpi4py
#pip install --no-cache-dir mpyq

#wget https://s3-us-west-2.amazonaws.com/ray-wheels/master/094748e73ac8d608d26ced0ed615ee0039b0e5d8/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl -O ray-3.0.0.dev0-cp311-cp311-linux_x86_64.whl

#wget https://files.pythonhosted.org/packages/4c/21/9ca93b84b92ef927814cb7ba37f0774a484c849d58f0b692b16af8eebcfb/pyarrow-17.0.0-cp311-cp311-manylinux_2_28_x86_64.whl -O pyarrow-17.0.0-cp311-cp311-linux_x86_64.whl
#pip install -U ray[rllib]

#pip install 'git+https://github.com/lcswillems/torch-ac.git'
#pip install 'git+https://github.com/IntelPython/mkl_fft.git'
#pip debug --verbose # to find compatible tags
#wget https://www.wheelodex.org/projects/mkl-fft/wheels/mkl_fft-1.3.8-70-cp310-cp310-manylinux2014_x86_64.whl
#mv mkl_fft-1.3.8-70-cp310-cp310-manylinux2014_x86_64.whl mkl_fft-1.3.8-70-cp310-cp310-linux_x86_64.whl
#pip install mkl_fft-1.3.8-72-cp310-cp310-linux_x86_64.whl 
#pip install cython
#pip install 'git+https://github.com/IntelPython/mkl_random.git'
#wget https://files.pythonhosted.org/packages/da/72/417f8e4807f0c7e83d708b27a152354132c0c967e469a72c1afc2207864a/mkl_random-1.2.4-92-cp310-cp310-manylinux2014_x86_64.whl
#mv mkl_random-1.2.4-92-cp310-cp310-manylinux2014_x86_64.whl mkl_random-1.2.4-92-cp310-cp310-linux_x86_64.whl
#pip install mkl_random-1.2.4-92-cp310-cp310-linux_x86_64.whl 
#wget https://files.pythonhosted.org/packages/d1/21/dd5cbe1a83d1b96fad3f808f33cf6e101491d7908b7409c62d89fb069706/mkl_service-2.4.1-0-cp310-cp310-manylinux2014_x86_64.whl
#mv mkl_service-2.4.1-0-cp310-cp310-manylinux2014_x86_64.whl mkl_service-2.4.1-0-cp310-cp310-linux_x86_64.whl
#pip install mkl_service-2.4.1-0-cp310-cp310-linux_x86_64.whl
# install this package first
# install on-policy package

cd /home/memole/projects/def-irina/memole/Lora
pip install -e .
#install starcraft
#mkdir 3rdparty
export SC2PATH="/home/memole/projects/def-irina/memole/Lora/3rdparty/StarCraftII"

#cd 3rdparty
#echo 'SC2PATH is set to '$SC2PATH
#wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula SC2.4.10.zip
#rm -rf SC2.4.10.zip
#export MAP_DIR="/content/drive/MyDrive/MPPO-ATTENTIOAN/3rdparty/StarCraftII/Maps/"
#echo 'MAP_DIR is set to '$MAP_DIR
#mkdir -p $MAP_DIR
#cd ..
#wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip SMAC_Maps.zip
#mv SMAC_Maps $MAP_DIR
#rm -rf SMAC_Maps.zip

#Hanabi

echo "Install Hanabi...."
#cd $DIR/onpolicy/envs/hanabi/
#cmake -B _build -S .
#cmake --build _build
#python -c 'import pyhanabi'

# install on-policy package
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
##install this package first
##football environment
#python3 -m pip install --upgrade pip setuptools psutil wheel
#pip install --no-index --no-cache-dir gfootball

# install melting pot
#pip install dm-env
#pip install pygame
##install DeepMind Lab2D https://github.com/deepmind/lab2d
#wget https://files.pythonhosted.org/packages/4b/31/884879224de4627b5d45b307cec8f4cd1e60db9aa61871e4aa2518c6584b/dmlab2d-1.0.0_dev.10-cp310-cp310-manylinux_2_31_x86_64.whl -O dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl
#setrpaths.sh --path dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
#pip install dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
#pip install dmlab2d
#pip install --no-index libcst
#git clone -b main https://github.com/deepmind/meltingpot
#cd meltingpot
#pip install --editable .[dev]
#pip install  --no-index --no-cache-dir dm-acme
wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
export CUDA_VISIBLE_DEVICES=0
#wandb sweep ./onpolicy/sweep_cleanup_config.yaml 
wandb agent zsheikhb/Lora/6owktc7t