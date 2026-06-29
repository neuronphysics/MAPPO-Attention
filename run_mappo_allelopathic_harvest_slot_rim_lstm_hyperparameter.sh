#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=140G  
#SBATCH --time=3-05:59:00
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/projects/def-irina/memole/logs/mappo-allelopathic_harvest-RIM-SLOT-parameter-search-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-irina/memole/logs/mappo-allelopathic_harvest-RIM-SLOT-parameter-search-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load python/3.11
module load scipy-stack/2024a
module load arrow/17.0.0
module load mujoco
module load openmpi
module load mpi4py/3.1.6
module load opencv/4.9.0
module load imkl/2023.2.0
module load rust/1.70.0
module load cmake

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
#pip install --no-cache-dir -U ray[rllib]

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

pip install -e .
#install starcraft
#mkdir 3rdparty
#export SC2PATH="/home/memole/projects/def-irina/memole/Lora/3rdparty/StarCraftII"

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

cd $CURRENT_PATH/onpolicy/scripts/train
# for QSA
echo "Train MAPPO model with SLOT, RIM and LSTM ..."


CUDA_VISIBLE_DEVICES=0 python3 train_meltingpot.py --use_valuenorm False --use_popart False --env_name "Meltingpot" --algorithm_name "mappo" --use_gae True \
     --experiment_name "allelopathic_harvest_RIM_SLOT_FIR_GRU_nu8_mgn05_lr00006" --substrate_name "allelopathic_harvest__open" --num_agents 16 --seed 42 --lr 0.00006 --critic_lr 0.00006 \
     --n_rollout_threads 24 --max_grad_norm 0.05 --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" --share_policy False \
     --use_centralized_V False --use_attention True --entropy_coef 0.3  --entropy_final_coef 0.003 --warmup_updates 0 --cooldown_updates 50000 --entropy_anneal_duration 100000 --attention_module "RIM" --rim_num_units 8 --rim_topk 6 \
     --hidden_size 192 --num_env_steps 4000000 --log_interval 1 --episode_length 2000 --downsample True --img_scale_factor 1 \
     --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 200 --slot_pretrain_batch_size 256 --rnn_attention_module "GRU" \
     --slot_att_work_path "/home/memole/scratch/meltingpot/slot_att/" --crop_size 88 \
     --slot_att_load_model True --use_slot_att True --use_pos_encoding True --use_input_att False --use_com_att True --use_x_reshape True \
     --slot_att_crop_repeat 2 --slot_log_fre 50 --collect_data False --no_train False --use_orthogonal True --value_loss_coef 0.75 --gain 0.01 --huber_delta 10 \
     --fine_tuning_type Partial --weight_decay 0.0001 --lr_main 0.00009 --use_orthogonal_loss True \
     --orthogonal_loss_coef 0.0025 --use_EWC False --perturb_interval 1456000 --num_iter 2 --ppo_epoch 10
