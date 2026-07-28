#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-22:59:59
#SBATCH --account=def-bengioy
#SBATCH --output=logs/ippo-train-bengio-allelopathic-harvest-rim-gru-seed-1_%N-%j.out
#SBATCH --error=logs/ippo-train-bengio-allelopathic-harvest-rim-gru-seed-1_%N-%j.err
##SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL

set -e
nvidia-smi

module --force purge

module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11
module load cuda
module load mujoco
module load scipy-stack/2024a
module load mpi4py/3.1.6
module load opencv/4.10.0
module load arrow/17.0.0
module load cmake

CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
#virtualenv --no-download --clear /home/memole/meltingpot
source /home/memole/meltingpot/bin/activate

REPO="$HOME/links/scratch/MAPPO-Attention"          # adjust to your checkout
JOB_ID="${SLURM_JOB_ID:-default}"

RESULTS="$REPO/onpolicy/scripts/results/Meltingpot"

#pip install --no-index --upgrade pip
#python -m pip install --upgrade pip setuptools wheel
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index --no-cache-dir torch torchvision torchtext torchaudio
#pip install --no-index --no-cache-dir wandb
#pip install  --no-index --no-cache-dir -r requirements.txt
#python -m pip install git+https://github.com/mpi4py/mpi4py
#pip install --no-cache-dir mpyq

#wget https://s3-us-west-2.amazonaws.com/ray-wheels/master/094748e73ac8d608d26ced0ed615ee0039b0e5d8/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl -O ray-3.0.0.dev0-cp311-cp311-linux_x86_64.whl
#python -m pip install "ray[rllib] @ file://$PWD/ray-3.0.0.dev0-cp311-cp311-linux_x86_64.whl"
#wget https://files.pythonhosted.org/packages/4c/21/9ca93b84b92ef927814cb7ba37f0774a484c849d58f0b692b16af8eebcfb/pyarrow-17.0.0-cp311-cp311-manylinux_2_28_x86_64.whl -O pyarrow-17.0.0-cp311-cp311-linux_x86_64.whl
#pip install -U ray[rllib]

#pip install 'git+https://github.com/lcswillems/torch-ac.git'
#pip install 'git+https://github.com/IntelPython/mkl_fft.git'
#pip debug --verbose # to find compatible tags
#wget https://files.pythonhosted.org/packages/4a/48/48d90c7cdad75d8205e54e233a382ecf2af2700b6ef7cad8bf25f85b253b/mkl_fft-1.3.8-72-cp311-cp311-manylinux2014_x86_64.whl -O mkl_fft-1.3.8-72-cp311-cp311-linux_x86_64.whl
#pip install mkl_fft-1.3.8-72-cp311-cp311-linux_x86_64.whl 


#wget https://mirrors.aliyun.com/pypi/packages/b8/0b/3c1b82099a0ddead8f0689aaa094506916a487096cc13ca96f7b514db228/mkl_service-2.5.0-1-cp311-cp311-manylinux_2_28_x86_64.whl -O mkl_service-2.5.0-1-cp311-cp311-linux_x86_64.whl
#pip install mkl_service-2.5.0-1-cp311-cp311-linux_x86_64.whl
#pip install 'git+https://github.com/IntelPython/mkl_random.git'
#wget https://files.pythonhosted.org/packages/b5/78/2da909eb0fa3d4973d5d47343afe726dd802314b6aef69ab41f6610b3638/mkl_random-1.2.4-92-cp311-cp311-manylinux2014_x86_64.whl -O mkl_random-1.2.4-92-cp311-cp311-linux_x86_64.whl
#pip install mkl_random-1.2.4-92-cp311-cp311-linux_x86_64.whl 

# install this package first
# install on-policy package

#pip install -e .
#install starcraft
#mkdir 3rdparty
#export SC2PATH="/home/memole/links/scratch/MAPPO-Attention/3rdparty/StarCraftII"

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
#python -m pip install dmlab2d
#pip install --no-index libcst
#git clone -b main https://github.com/deepmind/meltingpot
#cd meltingpot
#pip install --editable .[dev]
#pip install  --no-index --no-cache-dir dm-acme
#wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42

echo "Start running the train_meltingpot.py script ..."

export WANDB_MODE=offline
export WANDB_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb
export WANDB_CACHE_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb/cache
export WANDB_CONFIG_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb/config
#mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
EXP_NAME="allelopathic_harvest_rim_gru_trillium_bengio_mgn05_enco3_lr00006"
SUBSTRATE="allelopathic_harvest__open"
SCENARIO="collaborative_cooking__circuit_0"
ALGO="ippo"
CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm False --use_popart True --use_gae True --env_name "Meltingpot" --algorithm_name "$ALGO" \
     --experiment_name "$EXP_NAME" --substrate_name "$SUBSTRATE" --num_agents 16 --seed 123 --lr 0.00006 \
     --critic_lr 0.00006 --n_rollout_threads 22 --max_grad_norm 0.05 --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" \
     --entropy_coef 0.3 --entropy_final_coef 0.001 --warmup_updates 0 --cooldown_updates 50000 --entropy_anneal_duration 100000 \
     --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "GRU" --attention_module "RIM" --rim_num_units 8 \
     --rim_topk 6 --hidden_size 192 --num_env_steps 5000000 --log_interval 1 --episode_length 2000 --downsample True --img_scale_factor 8 \
     --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 250 --slot_pretrain_batch_size 80 \
     --slot_att_work_path "/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/slot_att/" \
     --slot_att_load_model False --use_slot_att False --use_pos_encoding True --use_input_att True --use_com_att True --use_x_reshape True \
     --slot_att_crop_repeat 9 --slot_log_fre 10 --collect_data False \
     --no_train False --use_orthogonal True --gain 0.01 --gamma 0.995 --clip_param 0.1 --gae_lambda 0.96

EXP_DIR="$RESULTS/$SCENARIO/$SUBSTRATE/$ALGO/$EXP_NAME"

MODEL_DIR=$(find "$EXP_DIR" -name "actor_agent_0.pt" -path "*${JOB_ID}*" -printf '%T@ %h\n' 2>/dev/null \
            | sort -rn | head -n1 | cut -d' ' -f2-)
if [[ -z "${MODEL_DIR}" ]]; then
    echo "No checkpoints under $EXP_DIR" >&2
    exit 1
fi
echo "MODEL_DIR=${MODEL_DIR}"
python3 -m onpolicy.scripts.render.render_meltingpot --env_name "Meltingpot" --algorithm_name "$ALGO" \
  --experiment_name "render_allelo_rim_gru" --substrate_name "$SUBSTRATE" \
  --use_render True --save_gifs True --n_rollout_threads 1 --episode_length 400 --render_episodes 1 --ifi 0.08 \
  --model_dir "$MODEL_DIR" --use_slot_att False \
  --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "GRU" \
  --attention_module "RIM" --rim_num_units 8 --rim_topk 6 --hidden_size 192 \
  --downsample True --img_scale_factor 8 --world_img_scale_factor 8 \
  --use_pos_encoding True --use_input_att True --use_com_att True --use_x_reshape True 