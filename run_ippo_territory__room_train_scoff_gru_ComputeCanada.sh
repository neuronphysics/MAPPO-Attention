#!/bin/bash
#SBATCH --job-name=IPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-13:59:59
#SBATCH --account=def-bengioy
#SBATCH --output=logs/ippo-train-bengio-territory-rooms-scoff-gru_%N-%j.out
#SBATCH --error=logs/ippo-train-bengio-territory-rooms-scoff-gru_%N-%j.err
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
source /home/memole/meltingpot/bin/activate

REPO="$HOME/links/scratch/MAPPO-Attention"
JOB_ID="${SLURM_JOB_ID:-default}"
RESULTS="$REPO/onpolicy/scripts/results/Meltingpot"

SEED="${1:-123}"
HIDDEN=300
SCOFF_NUM_UNITS="${SCOFF_NUM_UNITS:-6}"
SCOFF_TOPK="${SCOFF_TOPK:-5}"
SCOFF_NUM_SCHEMAS="${SCOFF_NUM_SCHEMAS:-4}"
RNN_CELL="${RNN_CELL:-GRU}"

if (( HIDDEN % SCOFF_NUM_UNITS != 0 )); then
    echo "hidden_size ($HIDDEN) must be divisible by SCOFF_NUM_UNITS ($SCOFF_NUM_UNITS)" >&2
    exit 2
fi
if (( SCOFF_TOPK < 0 || SCOFF_TOPK > SCOFF_NUM_UNITS )); then
    echo "SCOFF_TOPK must be between 0 and SCOFF_NUM_UNITS" >&2
    exit 2
fi
case "$RNN_CELL" in GRU|LSTM) ;; *) echo "RNN_CELL must be GRU or LSTM" >&2; exit 2 ;; esac

echo "Seed: $SEED | SCOFF: cell=$RNN_CELL units=$SCOFF_NUM_UNITS topk=$SCOFF_TOPK schemas=$SCOFF_NUM_SCHEMAS hidden=$HIDDEN"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline
export WANDB_DIR=$REPO/onpolicy/scripts/results/wandb
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_CONFIG_DIR=$WANDB_DIR/config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

EXP_NAME="territory_rooms_scoff_${RNN_CELL,,}_trillium_bengio_mgn0001_ent0004_lr00002_nu${SCOFF_NUM_UNITS}_tk${SCOFF_TOPK}_ns${SCOFF_NUM_SCHEMAS}"
SUBSTRATE="territory__rooms"
SCENARIO="collaborative_cooking__circuit_0"
ALGO="ippo"

CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm False --use_popart True --use_gae True \
     --env_name "Meltingpot" --algorithm_name "$ALGO" --experiment_name "$EXP_NAME" --substrate_name "$SUBSTRATE" --num_agents 9 --seed "$SEED" \
     --lr 0.00002 --critic_lr 0.00002 --n_rollout_threads 22 --max_grad_norm 0.001 \
     --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" \
     --entropy_coef 0.004 --entropy_final_coef 0.004 --warmup_updates 0 --cooldown_updates 50000 --entropy_anneal_duration 100000 \
     --share_policy False --use_centralized_V False --use_attention True \
     --rnn_attention_module "$RNN_CELL" --attention_module "SCOFF" \
     --scoff_num_units "$SCOFF_NUM_UNITS" --scoff_topk "$SCOFF_TOPK" --scoff_num_schemas "$SCOFF_NUM_SCHEMAS" --scoff_inp_heads 5 \
     --use_version_scoff 2 --scoff_do_relational_memory False --drop_out 0.0 \
     --hidden_size $HIDDEN --num_env_steps 4000000 --log_interval 1 --episode_length 1000 \
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
  --share_policy False --use_centralized_V False --use_attention True --rnn_attention_module "$RNN_CELL" \
  --attention_module "SCOFF" --scoff_num_units "$SCOFF_NUM_UNITS" --scoff_topk "$SCOFF_TOPK" --scoff_num_schemas "$SCOFF_NUM_SCHEMAS" \
  --use_version_scoff 1 --scoff_do_relational_memory False --drop_out 0.0 --hidden_size $HIDDEN \
  --downsample True --img_scale_factor 8 --world_img_scale_factor 8 \
  --use_pos_encoding False --use_input_att True --use_com_att True --use_x_reshape True

GIF_OUT="$REPO/onpolicy/scripts/results/gifs/${EXP_NAME}_${JOB_ID}.gif"
mkdir -p "$(dirname "$GIF_OUT")"
cp "$(dirname "$MODEL_DIR")/gifs/render.gif" "$GIF_OUT" && echo "gif -> $GIF_OUT"