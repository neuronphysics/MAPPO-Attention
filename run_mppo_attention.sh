#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01-13:59
#SBATCH --account=def-jhoey
#SBATCH --output=/home/memole/Montreal/on-policy/logs/MPPO-attention-seed-1_%N-%j.out
#SBATCH --error=/home/memole/Montreal/on-policy/logs/MPPO-attention-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda create -n marl python==3.8.10
conda activate marl
export WANDB_API_KEY ='a2a1bab96ebbc3869c65e3632485e02fcae9cc42'
export WANDB_ENTITY ='zsheikhb'
pip install torch==2.0.1+cu118 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# install this package first
pip install --no-index seaborn
pip install --no-index absl-py 
pip install --no-index gym 
pip install --no-index wandb
pip install --no-index tensorboardX 
pip install --no-index torch-ac
pip install --no-index imageio 
pip install --no-index pyglet 
pip install --no-index PIL
# install on-policy package

pip install -e .

#install starcraft
#based on https://github.com/oxwhirl/pymarl/tree/master
bash install_sc2.sh
#Hanabi
pip install --no-index cffi
cd /home/memole/Montreal/MPPO-ATTENTIOAN/onpolicy/envs/hanabi
mkdir build & cd build
cmake ..
make -j
# install this package first
#football environment
apt-get update
apt-get install libsdl2-gfx-dev libsdl2-ttf-dev

# 2.8 and 2.9 binary is the same, so we use 2.8 .so file
#https://github.com/google-research/football/blob/master/README.md
git clone -b v2.9 https://github.com/google-research/football.git
mkdir -p football/third_party/gfootball_engine/lib

wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
cd football && GFOOTBALL_USE_PREBUILT_SO=1 python3 -m pip install .

cd /home/memole/Montreal/MPPO-ATTENTIOAN/onpolicy/scripts/train_mpe_scripts
chmod +x train_mpe_comm.sh
bash train_mpe_comm.sh