#!/bin/bash

#SBATCH --cpus-per-task=18                                # Ask for 10 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1                                    # Ask for 1 GPU
#SBATCH --constraint="dgx&ampere"
#SBATCH --mem=100G                                        # Ask for 10 GB of RAM
#SBATCH --time=2-23:23:59                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-allelopathic_harvest-train-RIM-LSTM-%j.out  # Write the log on scratch
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/logs/slurm-allelopathic_harvest-train-RIM-LSTM-%j.err  # Write the err on scratch

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
echo "Train MAPPO model with RIM and LSTM ..."



CUDA_VISIBLE_DEVICES=0 python3 -m onpolicy.scripts.train.train_meltingpot --use_valuenorm False --use_popart False --env_name "Meltingpot" --algorithm_name "mappo" \
    --use_gae True --experiment_name "allelopathic_harvest_RIM_LSTM_mgn04_lr00006_nu8_tk_6" --substrate_name "allelopathic_harvest__open" --num_agents 16 \
    --seed 42 --lr 0.00006 --critic_lr 0.00006 --n_rollout_threads 24 --max_grad_norm 0.04 --use_wandb True --user_name "irina-rish" --wandb_name "irina-rish" \
    --share_policy False --use_centralized_V False --use_attention True --entropy_coef 0.12 --entropy_final_coef 0.004 --warmup_updates 50000 --cooldown_updates 50000 \
    --entropy_anneal_duration 800000 --attention_module "RIM" --rim_num_units 8 --rim_topk 6 --hidden_size 192 --num_env_steps 4000000 --log_interval 1 --episode_length 2000 \
    --downsample True --img_scale_factor 8 --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 200 --slot_pretrain_batch_size 200 --rnn_attention_module "LSTM" \
    --slot_att_work_path "/home/mila/z/zahra.sheikhbahaee/scratch/main/onpolicy/scripts/results/slot_att/" \
    --slot_att_load_model False --use_slot_att False --use_pos_encoding True --use_input_att True --use_com_att True \
    --use_x_reshape True --slot_att_crop_repeat 2 --slot_log_fre 50 --collect_data False --no_train False --use_orthogonal True \
    --gain 0.01 --drop_out 0.2 --gamma 0.995 --clip_param 0.2 --gae_lambda 0.97