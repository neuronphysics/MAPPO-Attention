#!/bin/bash
#SBATCH --job-name=IPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-20:59:59
#SBATCH --account=def-irina
#SBATCH --output=logs/ippo-train-irina-prisoners-dilemma-rim-gru_%N-%j.out
#SBATCH --error=logs/ippo-train-irina-prisoners-dilemma-rim-gru_%N-%j.err
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
source /home/memole/meltingpot/bin/activate

REPO="$HOME/links/scratch/MAPPO-Attention"          # adjust to your checkout
JOB_ID="${SLURM_JOB_ID:-default}"

RESULTS="$REPO/onpolicy/scripts/results/Meltingpot"

SEED="${1:-123}"
HIDDEN=300
RIM_NUM_UNITS="${RIM_NUM_UNITS:-6}"
RIM_TOPK="${RIM_TOPK:-5}"
RNN_CELL="GRU"

if (( HIDDEN % RIM_NUM_UNITS != 0 )); then
    echo "hidden_size ($HIDDEN) should be divisible by RIM_NUM_UNITS ($RIM_NUM_UNITS)" >&2
    exit 2
fi
if (( RIM_TOPK < 0 || RIM_TOPK > RIM_NUM_UNITS )); then
    echo "RIM_TOPK must be between 0 and RIM_NUM_UNITS" >&2
    exit 2
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# NOTE: 22 rollout workers on 16 CPUs oversubscribes slightly (same as the
# allelopathic run). Raise --cpus-per-task to 22 if env stepping is the
# bottleneck on this substrate.

echo "Start running the train_meltingpot.py script ..."
echo "Seed: $SEED | RIM: cell=$RNN_CELL units=$RIM_NUM_UNITS topk=$RIM_TOPK hidden=$HIDDEN | threads=22"

# W&B auth must come from WANDB_API_KEY or a prior `wandb login`.
# NEVER put an API key in this script. Offline mode below needs no auth.
export WANDB_MODE=offline
export WANDB_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb
export WANDB_CACHE_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb/cache
export WANDB_CONFIG_DIR=/home/memole/links/scratch/MAPPO-Attention/onpolicy/scripts/results/wandb/config
#mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

EXP_NAME="prisoners_dilemma_rim_${RNN_CELL,,}_nu${RIM_NUM_UNITS}_tk${RIM_TOPK}_enc03_seed${SEED}"
SUBSTRATE="prisoners_dilemma_in_the_matrix__arena"
ALGO="ippo"

CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm False --use_popart True --env_name "Meltingpot" --algorithm_name "$ALGO" \
     --experiment_name "$EXP_NAME" --substrate_name "$SUBSTRATE" --num_agents 8 --seed "$SEED" --lr 0.00009 \
     --critic_lr 0.000095 --n_rollout_threads 22 --max_grad_norm 0.075 --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" \
     --entropy_coef 0.3 --entropy_final_coef 0.004 --warmup_updates 0 --cooldown_updates 50000 --entropy_anneal_duration 100000 \
     --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "$RNN_CELL" --attention_module "RIM" \
     --rim_num_units "$RIM_NUM_UNITS" --rim_topk "$RIM_TOPK" --hidden_size $HIDDEN \
     --num_env_steps 5000000 --log_interval 1 --episode_length 1000 --gae_lambda 0.97 --drop_out 0.2 --gamma 0.995 \
     --downsample True --img_scale_factor 8 --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 200 \
     --slot_pretrain_batch_size 200 \
     --slot_att_work_path "$REPO/onpolicy/scripts/results/slot_att/" \
     --slot_att_load_model False --use_slot_att False --use_pos_encoding True --use_input_att True --use_com_att True --use_x_reshape True \
     --slot_att_crop_repeat 2 --slot_log_fre 50 --collect_data False \
     --no_train False --value_loss_coef 0.5 --gain 0.04 --huber_delta 10 --use_orthogonal True --clip_param 0.5

# Locate the newest checkpoint for this experiment/job anywhere under RESULTS
# (robust to the scenario/substrate directory layout).
MODEL_DIR=$(find "$RESULTS" -name "actor_agent_0.pt" -path "*${EXP_NAME}*" -path "*${JOB_ID}*" -printf '%T@ %h\n' 2>/dev/null \
            | sort -rn | head -n1 | cut -d' ' -f2-)
if [[ -z "${MODEL_DIR}" ]]; then
    MODEL_DIR=$(find "$RESULTS" -name "actor_agent_0.pt" -path "*${EXP_NAME}*" -printf '%T@ %h\n' 2>/dev/null \
                | sort -rn | head -n1 | cut -d' ' -f2-)
fi
if [[ -z "${MODEL_DIR}" ]]; then
    echo "No checkpoints under $RESULTS for $EXP_NAME" >&2
    exit 1
fi
echo "MODEL_DIR=${MODEL_DIR}"
python3 -m onpolicy.scripts.render.render_meltingpot --env_name "Meltingpot" --algorithm_name "$ALGO" \
  --experiment_name "render_pd_rim_${RNN_CELL,,}" --substrate_name "$SUBSTRATE" \
  --use_render True --save_gifs True --n_rollout_threads 1 --episode_length 400 --render_episodes 1 --ifi 0.08 \
  --model_dir "$MODEL_DIR" --use_slot_att False \
  --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "$RNN_CELL" \
  --attention_module "RIM" --rim_num_units "$RIM_NUM_UNITS" --rim_topk "$RIM_TOPK" --hidden_size $HIDDEN --drop_out 0.2 \
  --downsample True --img_scale_factor 8 --world_img_scale_factor 8 \
  --use_pos_encoding True --use_input_att True --use_com_att True --use_x_reshape True