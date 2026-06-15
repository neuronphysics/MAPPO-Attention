#!/bin/bash

#SBATCH --cpus-per-task=16                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1                                     # Ask for 1 GPU
#SBATCH --constraint="dgx&ampere"
#SBATCH --mem=120G                                        # Ask for 10 GB of RAM
#SBATCH --time=2-03:59:59                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-allelopathic_harvest-train-QSA-rim-%j.out  # Write the log on scratch
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-allelopathic_harvest-train-QSA-rim-%j.err  # Write the err on scratch

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
echo "PPO with slot attention QSA and RIM"
CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm True --use_popart True --use_gae True --env_name "Meltingpot" --algorithm_name "mappo" \
     --experiment_name "allelopathic_harvest_mila_nu8_slot_attention_QSA_RIM_LSTM" --substrate_name "allelopathic_harvest__open" --num_agents 16 --seed 999 --lr 0.00006 --critic_lr 0.00007 --n_rollout_threads 20 \
	 --max_grad_norm 0.075 --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" --share_policy False --use_centralized_V False --load_model False --use_attention True --entropy_coef 0.02 \
	 --entropy_final_coef 0.005 --warmup_updates 200000 --cooldown_updates 300000 --entropy_anneal_duration 2500000 --attention_module "RIM" --rnn_attention_module "LSTM" --rim_num_units 8 --rim_topk 4 \
	 --hidden_size 192 --num_env_steps 4000000 --log_interval 1 --episode_length 2000 --downsample True --img_scale_factor 1 --world_img_scale_factor 8 \
	 --slot_att_work_path "/home/mila/z/zahra.sheikhbahaee/scratch/main/onpolicy/scripts/results/slot_att/" --slot_att_load_model True --pretrain_slot_att False \
	 --slot_train_ep 200 --slot_pretrain_batch_size 150 --use_slot_att True --use_pos_encoding True --use_input_att False --use_com_att True --use_x_reshape True \
	 --slot_att_crop_repeat 9 --slot_log_fre 10 --no_train False --collect_data False --collect_agent False --collect_world False --collect_data_ep_num 20 \
	 --crop_size 88 --value_loss_coef 0.75 --gain 0.01 --grad_clip 0.2 --clip_param 0.1 --use_orthogonal True --fine_tuning_type "Partial" --weight_decay 0.0001 \
	 --lr_main 0.00009 --use_orthogonal_loss True --orthogonal_loss_coef 0.1 --use_EWC False --perturb_interval 1456000 --num_iter 2



