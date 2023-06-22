#!/bin/bash
#SBATCH --job-name=MPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=95000M   
#SBATCH --time=1-20:00:00
#SBATCH --account=def-gdumas85
#SBATCH --output=~/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MPPO-attention-seed-1_%N-%j.out
#SBATCH --error=~/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MPPO-attention-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load gcc python/3.10 opencv
#virtualenv --no-download --clear ~/Montreal/marl_envs

git clone https://neuronphysics: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@github.com/neuronphysics/MPPO-ATTENTIOAN.git
virtualenv --no-download --clear $SLURM_TMPDIR
source $SLURM_TMPDIR/bin/activate
cd MPPO-ATTENTIOAN/
PATH=`pwd`
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchtext torchaudio

pip install -r requirements.txt
python -m pip install git+https://github.com/mpi4py/mpi4py
pip install --no-cache-dir mpyq
pip install --no-cache-dir mujoco-py
pip install --no-index --no-cache-dir torch-ac

pip install 'git+https://github.com/IntelPython/mkl_fft.git'
pip install 'git+https://github.com/IntelPython/mkl_random.git'
wandb login $API_KEY
# install this package first

# install on-policy package
pip install -e .

#install starcraft
export SC2PATH='/home/memole/Montreal/StarCraftII'
#Hanabi


cd $PATH/onpolicy/envs/hanabi/
cmake -B _build -S .
cmake --build _build
python -c 'import pyhanabi'
##install this package first
##football environment
#python3 -m pip install --upgrade pip setuptools psutil wheel
#python3 -m pip install gfootball

cd $PATH/onpolicy/scripts/train_mpe_scripts
chmod +x train_mpe_comm.sh
bash train_mpe_comm.sh