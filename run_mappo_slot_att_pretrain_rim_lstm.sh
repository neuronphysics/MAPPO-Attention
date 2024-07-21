#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G  
#SBATCH --time=2-10:00:00
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/projects/def-irina/memole/logs/mappo-attention-pretrain-slot_att-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-irina/memole/logs/mappo-attention-pretrain-slot_att-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc python/3.10 mujoco cuda
module load mpi4py/3.1.6
module spider pyarrow/17.0.0
module load opencv/4.10.0
module load scipy-stack
module spider mkl
module spider rust/1.65.0
module load cmake
DIR=/home/memole/projects/def-irina/memole/LSTM

#virtualenv --no-download --clear /home/memole/meltingpot
source /home/memole/meltingpot/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --no-index --upgrade pip
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index torch torchvision torchtext torchaudio
#pip install --no-index --no-cache-dir wandb
#pip install --no-cache-dir -r ~/projects/def-irina/memole/LSTM/requirements.txt
#python -m pip install git+https://github.com/mpi4py/mpi4py
#pip install --no-cache-dir mpyq

#wget https://s3-us-west-2.amazonaws.com/ray-wheels/master/094748e73ac8d608d26ced0ed615ee0039b0e5d8/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl -O ray-3.0.0.dev0-cp311-cp311-linux_x86_64.whl

#wget https://files.pythonhosted.org/packages/4c/21/9ca93b84b92ef927814cb7ba37f0774a484c849d58f0b692b16af8eebcfb/pyarrow-17.0.0-cp311-cp311-manylinux_2_28_x86_64.whl -O pyarrow-17.0.0-cp311-cp311-linux_x86_64.whl
#pip install -U ray[rllib]

#pip install 'git+https://github.com/lcswillems/torch-ac.git'
#pip install 'git+https://github.com/IntelPython/mkl_fft.git'
#pip debug --verbose # to find compatible tags
#wget https://files.pythonhosted.org/packages/4a/48/48d90c7cdad75d8205e54e233a382ecf2af2700b6ef7cad8bf25f85b253b/mkl_fft-1.3.8-72-cp311-cp311-manylinux2014_x86_64.whl
#mv mkl_fft-1.3.8-72-cp311-cp311-manylinux2014_x86_64.whl mkl_fft-1.3.8-72-cp311-cp311-linux_x86_64.whl
#pip install mkl_fft-1.3.8-72-cp311-cp311-linux_x86_64.whl 

#pip install 'git+https://github.com/IntelPython/mkl_random.git'
#wget https://files.pythonhosted.org/packages/b5/78/2da909eb0fa3d4973d5d47343afe726dd802314b6aef69ab41f6610b3638/mkl_random-1.2.4-92-cp311-cp311-manylinux2014_x86_64.whl
#mv mkl_random-1.2.4-92-cp311-cp311-manylinux2014_x86_64.whl mkl_random-1.2.4-92-cp311-cp311-linux_x86_64.whl
#pip install mkl_random-1.2.4-92-cp311-cp311-linux_x86_64.whl 

# install this package first
# install on-policy package

cd /home/memole/projects/def-irina/memole/LSTM
pip install -e .
#install starcraft
#mkdir 3rdparty
export SC2PATH="/home/memole/projects/def-irina/memole/LSTM/3rdparty/StarCraftII"

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
echo "Start running the train_meltingpot.py script ..."
cd $DIR/onpolicy/scripts/train
#CUDA_VISIBLE_DEVICES=0,1 python train_mpe.py --use_valuenorm --use_popart --env_name "MPE" --algorithm_name "mappo" --experiment_name "check" \
#    --scenario_name "simple_speaker_listener" --num_agents 2 --num_landmarks 3 --seed 1 --use_render \
#    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 \
#    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy

CUDA_VISIBLE_DEVICES=0 python3 train_meltingpot.py --use_valuenorm False --use_popart True --env_name "Meltingpot" --algorithm_name "mappo" \
     --experiment_name "pretrain_qsa_rim" --substrate_name "territory__rooms" --num_agents 9 --seed 123 --lr 0.00002 \
     --critic_lr 0.00002 --n_rollout_threads 1 --max_grad_norm 0.01 --use_wandb False --user_name "zsheikhb" --wandb_name "zsheikhb" \
     --share_policy False --use_centralized_V False --use_attention True --entropy_coef 0.004 --attention_module "RIM" --rim_num_units 6 \
     --rim_topk 4 --hidden_size 300 --num_env_steps 4000000 --log_interval 1 --episode_length 1000 --downsample True --img_scale_factor 1 \
     --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 150 --slot_pretrain_batch_size 100 \
     --slot_att_work_path "/home/memole/projects/def-irina/memole/LSTM/onpolicy/scripts/results/slot_att/" \
     --slot_att_load_model False --use_slot_att False --use_pos_encoding False --use_input_att False --use_com_att True --use_x_reshape True \
     --slot_att_crop_repeat 9 --slot_log_fre 1 --collect_data True --collect_agent False --collect_world True --collect_data_ep_num 25 \
     --no_train True --crop_size 88