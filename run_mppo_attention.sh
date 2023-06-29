#!/bin/bash
#SBATCH --job-name=MPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=95000M   
#SBATCH --time=1-10:00:00
#SBATCH --account=def-gdumas85
#SBATCH --output=/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MPPO-attention-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MPPO-attention-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc python/3.10 opencv/4.7
module load scipy-stack
#virtualenv --no-download --clear ~/Montreal/marl_envs
cd /scratch/memole
DIR=/scratch/memole/MPPO-ATTENTIOAN

virtualenv --no-download --clear $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir numpy 
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index wandb
pip install -r $DIR/requirements.txt
python -m pip install git+https://github.com/mpi4py/mpi4py
pip install --no-cache-dir mpyq
pip install --no-cache-dir mujoco-py


pip install 'git+https://github.com/lcswillems/torch-ac.git'
pip install 'git+https://github.com/IntelPython/mkl_fft.git'
pip install 'git+https://github.com/IntelPython/mkl_random.git'
wandb login $API_KEY
# install this package first

#install starcraft
export SC2PATH='/home/memole/Montreal/StarCraftII'
#Hanabi

echo "Install Hanabi...."
cd $DIR/onpolicy/envs/hanabi/
cmake -B _build -S .
cmake --build _build
python -c 'import pyhanabi'

# install on-policy package
echo "Install MPPO ...."
cd $DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
pip install -e .
##install this package first
##football environment
#python3 -m pip install --upgrade pip setuptools psutil wheel
#python3 -m pip install gfootball
echo "Start running the train_mpe_comm.sh script ..."
cd $DIR/onpolicy/scripts/train_mpe_scripts
chmod +x train_mpe_comm.sh
bash train_mpe_comm.sh