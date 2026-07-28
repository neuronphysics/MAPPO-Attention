#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-23:59:59
#SBATCH --account=def-bengioy
#SBATCH --output=logs/mappo-train-bengio-territory-rooms-rim-lstm-seed-1_%N-%j.out
#SBATCH --error=logs/mappo-train-bengio-territory-rooms-rim-lstm-seed-1_%N-%j.err
#SBATCH --mail-type=END,FAIL

set -e
nvidia-smi
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11 cuda mujoco scipy-stack/2024a mpi4py/3.1.6 opencv/4.10.0 arrow/17.0.0 cmake

source /home/memole/meltingpot/bin/activate

REPO="$HOME/links/scratch/MAPPO-Attention"
JOB_ID="${SLURM_JOB_ID:-default}"
RESULTS="$REPO/onpolicy/scripts/results/Meltingpot"

# (… keep your commented install block here if you want the reference …)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline
export WANDB_DIR=$REPO/onpolicy/scripts/results/wandb
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_CONFIG_DIR=$WANDB_DIR/config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

EXP_NAME="territory_rooms_rim_lstm_trillium_bengio_mgn0001_ent0004_lr00002_nu6_tk5"
SUBSTRATE="territory__rooms"
SCENARIO="collaborative_cooking__circuit_0"
ALGO="ippo"

CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm False --use_popart True --use_gae True \
     --env_name "Meltingpot" --algorithm_name "$ALGO" --experiment_name "$EXP_NAME" --substrate_name "$SUBSTRATE" --num_agents 9 --seed 123 \
     --lr 0.00002 --critic_lr 0.00002 --n_rollout_threads 22 --max_grad_norm 0.001 \
     --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" \
     --entropy_coef 0.004 --entropy_final_coef 0.004 --warmup_updates 0 --cooldown_updates 50000 --entropy_anneal_duration 100000 \
     --share_policy False --use_centralized_V False --use_attention True \
     --rnn_attention_module "LSTM" --attention_module "RIM" --rim_num_units 6 --rim_topk 5 \
     --hidden_size 300 --num_env_steps 4000000 --log_interval 1 --episode_length 1000 \
     --downsample True --img_scale_factor 8 --world_img_scale_factor 8 \
     --pretrain_slot_att False --slot_train_ep 200 --slot_pretrain_batch_size 200 \
     --slot_att_work_path "$REPO/onpolicy/scripts/results/slot_att/" \
     --slot_att_load_model False --use_slot_att False \
     --use_pos_encoding False --use_input_att True --use_com_att True --use_x_reshape True \
     --no_train False --use_orthogonal True --gain 0.01 --value_loss_coef 0.5 

EXP_DIR="$RESULTS/$SCENARIO/$SUBSTRATE/$ALGO/$EXP_NAME"
MODEL_DIR=$(find "$EXP_DIR" -name "actor_agent_0.pt" -path "*${JOB_ID}*" -printf '%T@ %h\n' 2>/dev/null \
            | sort -rn | head -n1 | cut -d' ' -f2-)
if [[ -z "${MODEL_DIR}" ]]; then
    echo "No checkpoints under $EXP_DIR" >&2
    exit 1
fi
echo "MODEL_DIR=${MODEL_DIR}"

python3 -m onpolicy.scripts.render.render_meltingpot --env_name "Meltingpot" --algorithm_name "$ALGO" \
  --experiment_name "$EXP_NAME" --substrate_name "$SUBSTRATE" --num_agents 9 \
  --use_render True --save_gifs True --n_rollout_threads 1 --episode_length 400 --render_episodes 1 --ifi 0.08 \
  --model_dir "$MODEL_DIR" \
  --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "LSTM" \
  --attention_module "RIM" --rim_num_units 6 --rim_topk 5 --hidden_size 300 \
  --downsample True --img_scale_factor 8 --world_img_scale_factor 8 \
  --use_pos_encoding False --use_input_att True --use_com_att True --use_x_reshape True

GIF_OUT="$REPO/onpolicy/scripts/results/gifs/${EXP_NAME}_${JOB_ID}.gif"
mkdir -p "$(dirname "$GIF_OUT")"
cp "$(dirname "$MODEL_DIR")/gifs/render.gif" "$GIF_OUT" && echo "gif -> $GIF_OUT"