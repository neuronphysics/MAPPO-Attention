#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G  
#SBATCH --time=6-23:59:00
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/projects/def-gdumas85/memole/Meltingpot/logs/mappo-territory__room_train-slot-attention-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-gdumas85/memole/Meltingpot/logs/mappo-territory__room_train-slot-attention-seed-1_%N-%j.err
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
#install mappo
cd $CURRENT_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
pip install -e .
#install starcraft
#mkdir 3rdparty
export SC2PATH="/home/memole/projects/def-gdumas85/memole/Meltingpot/slow-unfreeze/3rdparty/StarCraftII"

CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"

cd $CURRENT_PATH/onpolicy/scripts/train
# for QSA
echo "Train MAPPO model with LSTM ..."


CUDA_VISIBLE_DEVICES=0 python3 train_meltingpot.py --use_valuenorm False --use_popart True --env_name "Meltingpot" --algorithm_name "mappo" \
   --experiment_name "train_territory_room_slot_qsa_rim_lstm"  --use_gae True --substrate_name "territory__rooms" --num_agents 9 --seed 123 --lr 0.00005 --critic_lr 0.00005 \
   --n_rollout_threads 15 --max_grad_norm 0.05 --use_wandb False --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy False \
   --use_centralized_V False --load_model False --use_attention True --entropy_coef 0.01 --entropy_final_coef 0.005 --warmup_updates 100000 --cooldown_updates 100000 --entropy_anneal_duration 700000 --attention_module "RIM" --rim_num_units 6 --rim_topk 4 --hidden_size 204 \
   --num_env_steps 4000000 --log_interval 3 --episode_length 1000 --downsample True --img_scale_factor 1 --rnn_attention_module "LSTM" \
   --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 150 --slot_pretrain_batch_size 400 \
   --slot_att_work_path "/home/memole/projects/def-gdumas85/memole/Meltingpot/pretrain_slot_att/" --slot_att_load_model True \
   --use_slot_att True --use_pos_encoding True --use_input_att False --use_com_att True --use_x_reshape True --slot_att_crop_repeat 9 --slot_log_fre 20 \
   --collect_data False --collect_agent False --collect_world True --collect_data_ep_num 20 --no_train False --crop_size 88 \
   --value_loss_coef 0.75 --gain 0.01 --clip_param 0.1 --grad_clip 0.2 --use_orthogonal True --fine_tuning_type "Slowly_Unfreeze" --lr_main 0.00009 --unfreeze_episode 13