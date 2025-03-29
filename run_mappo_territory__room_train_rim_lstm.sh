#!/bin/bash

#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1                                    # Ask for 1 GPU
#SBATCH --constraint="dgx&ampere"
#SBATCH --mem=80G                                        # Ask for 10 GB of RAM
#SBATCH --time=2-20:40:59                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-territory_rooms-train-RIM-LSTM-%j.out  # Write the log on scratch
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-territory_rooms-train-RIM-LSTM-%j.err  # Write the err on scratch

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
module unload python
module load anaconda/3
conda activate meltingpot
module load gcc/9.3.0
module unload anaconda
module load python/3.10

CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install -e .
seed=$1

echo "seed ---> $seed"




wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
echo "Start running the train meltingpot script ..."

CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"

# for QSA
echo "Train MAPPO model with LSTM ..."


CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm True --use_popart True --use_gae True --env_name "Meltingpot" --algorithm_name "mappo" \
     --experiment_name "territory_rooms_nu6_tk5_RIM_LSTM" --substrate_name "territory__rooms" --num_agents 9 --seed 999 --lr 0.00005 --critic_lr 0.00005 \
     --n_rollout_threads 9 --max_grad_norm 0.075 --use_wandb True --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy False --use_centralized_V False --load_model False \
     --use_attention True --entropy_coef 0.01 --entropy_final_coef 0.005 --warmup_updates 100000 --cooldown_updates 100000 --entropy_anneal_duration 600000 --attention_module "RIM" \
     --rnn_attention_module "LSTM" --rim_num_units 6 --rim_topk 4 --hidden_size 270 --num_env_steps 4000000 --log_interval 1 --episode_length 1000 --downsample True \
     --img_scale_factor 8 --world_img_scale_factor 8 --slot_att_work_path "/home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/LSTM/onpolicy/scripts/results/slot_att/" --slot_att_load_model True \
     --pretrain_slot_att False --slot_train_ep 200 --slot_pretrain_batch_size 1000 --use_slot_att False --use_pos_encoding True --use_input_att False --use_com_att True \
     --use_x_reshape True --slot_att_crop_repeat 9 --slot_log_fre 10 --no_train False --collect_data False --collect_agent False --collect_world False --collect_data_ep_num 20 \
     --crop_size 88 --value_loss_coef 0.75 --gain 0.01 --grad_clip 0.2 --clip_param 0.1 --use_orthogonal True --fine_tuning_type "Lora" --weight_decay 0.0005 