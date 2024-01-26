#!/bin/bash
SBATCH --job-name=MP_dane
SBATCH --nodes=1
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=9
SBATCH --gres=gpu:v100:4
SBATCH --mem=16G
SBATCH --time=20:00:00
SBATCH --mail-user=dane.malenfant@mila.quebec              # notification for job conditions
SBATCH --mail-type=END
SBATCH --output=home/mila/d/dane.malenfant/MAPPO_ATT/test.out


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
#pip install --no-index --upgrade pip
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index torch torchvision torchtext torchaudio
#pip install --no-index wandb
#pip install --no-cache-dir -r ~/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/requirements.txt
#python -m pip install git+https://github.com/mpi4py/mpi4py
#pip install --no-cache-dir mpyq
module load python/3.10
source /home/mila/d/dane.malenfant/MAPPO_ATT/att/bin/activate
pip install -r requirements.txt

#pip install 'git+https://github.com/lcswillems/torch-ac.git'
#pip install 'git+https://github.com/IntelPython/mkl_fft.git'
#pip install 'git+https://github.com/IntelPython/mkl_random.git'


# install this package first
# install on-policy package

#cd /home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/
#pip install -e .
#install starcraft
#mkdir 3rdparty
#export SC2PATH="/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/3rdparty/StarCraftII"

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

#echo "Install Hanabi...."
#cd $DIR/onpolicy/envs/hanabi/
#cmake -B _build -S .
#cmake --build _build
#python -c 'import pyhanabi'

# install on-policy package
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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
#pip install --no-index libcst
#git clone -b main https://github.com/deepmind/meltingpot
#cd meltingpot
#pip install --editable .[dev]
#pip install  --no-index --no-cache-dir dm-acme
wandb login 312073ec61b7abf9e385955c376fdeccc55d1086
echo "Start running the train_mpe_comm.sh script ..."

python "/MAPPO_ATT/local_test.py" --hidden_size 72 --use_attention True --env_name 'Meltingpot' --substrate_name 'territory__rooms' --num_agents 9 --algorithm_name 'mappo' --seed 1 --lr 2e-4 > dane.log
