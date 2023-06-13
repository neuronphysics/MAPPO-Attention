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

pip install torch==2.0.1+cu118 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# install this package first
pip install seaborn
pip install absl-py 
pip install gym 
pip install wandb 
pip install tensorboardX 
pip install torch-ac
pip install imageio 
pip install pyglet 
pip install PIL
# install on-policy package

pip install -e .

#install starcraft
#based on https://github.com/oxwhirl/pymarl/tree/master
bash install_sc2.sh
#Hanabi
pip install cffi
cd /home/memole/Montreal/on-policy/onpolicy/envs/hanabi
mkdir build & cd build
cmake ..
make -j
# install this package first
pip install seaborn
#football environment
apt-get update
apt-get install libsdl2-gfx-dev libsdl2-ttf-dev

# 2.8 and 2.9 binary is the same, so we use 2.8 .so file
#https://github.com/google-research/football/blob/master/README.md
git clone -b v2.9 https://github.com/google-research/football.git
mkdir -p football/third_party/gfootball_engine/lib

wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
cd football && GFOOTBALL_USE_PREBUILT_SO=1 python3 -m pip install .

cd /home/memole/Montreal/on-policy/onpolicy/scripts/train_mpe_scripts
chmod +x train_mpe_comm.sh
bash train_mpe_comm.sh