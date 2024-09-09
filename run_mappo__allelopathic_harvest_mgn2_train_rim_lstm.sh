#!/bin/bash

#SBATCH --cpus-per-task=6                                # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1                                    # Ask for 1 GPU
#SBATCH --constraint="dgx&ampere"
#SBATCH --mem=80G                                        # Ask for 10 GB of RAM
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

cd $CURRENT_PATH/onpolicy/scripts/train
# for QSA
echo "Train MAPPO model with LSTM ..."


CUDA_VISIBLE_DEVICES=0 python3 train_meltingpot.py --use_valuenorm False --use_popart True --env_name "Meltingpot" --algorithm_name "mappo" \
     --experiment_name "harvest_rim_LSTM_nu6_tk_4_mgn2_EntCoef_004_lr00003" --substrate_name "allelopathic_harvest__open" --num_agents 16 --seed 123 --lr 0.00003 --critic_lr 0.00003 \
     --n_rollout_threads 4 --max_grad_norm 0.2 --use_wandb True --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy False \
     --use_centralized_V False --use_attention True --entropy_coef 0.004 --attention_module "RIM" --rim_num_units 6 --rim_topk 4 \
     --hidden_size 300 --num_env_steps 4000000 --log_interval 1 --episode_length 1000 --downsample True --img_scale_factor 8 \
     --world_img_scale_factor 8 --pretrain_slot_att False --slot_train_ep 200 --slot_pretrain_batch_size 200 --rnn_attention_module "LSTM" \
     --slot_att_work_path "/home/mila/z/zahra.sheikhbahaee/Projects/meltingpot/LSTM/onpolicy/scripts/results/slot_att/" \
     --slot_att_load_model False --use_slot_att False --use_pos_encoding False --use_input_att True --use_com_att True --use_x_reshape True \
     --slot_att_crop_repeat 2 --slot_log_fre 50 --collect_data False --no_train False