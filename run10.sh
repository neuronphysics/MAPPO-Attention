#!/bin/bash

echo Running on $HOSTNAME

module unload python
module load anaconda/3
conda activate ~/MAPPO_ATT/att;
module load gcc/9.3.0
module unload anaconda
module load python/3.10


seed=$1


#python -u $algorithm  --algo Conspec --env-name "MontezumaRevengeNoFrameskip-v4" --lrCL 20e-4   --choiceCLparams 0 --seed $seed --head 1 --algo ppoCL --use-gae --lr 2e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 6 --num-steps 1200 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --skip 4

#wandb login 312073ec61b7abf9e385955c376fdeccc55d1086
echo "Start running the train_mpe_comm.sh script ..."

python local_test.py --hidden_size 72  --env_name 'Meltingpot' --substrate_name 'allelopathic_harvest__open' --experiment_name "harvest_decoder_optimizer_LSTM_wLR_KL" --num_agents 16 --algorithm_name 'mappo'  --seed $seed --lr 0.0007 --num_env_steps 500000 --episode_length 2000 

#python -u $file --pyco 1 --seed $seed --expansion 24 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 2.0  --head 8

#python -u $file --seed $seed --expansion 62 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR 0.2 --head 8


#python -u $algo  --algo ppoCLsep --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 89 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --env-name Skiing-v0 --lrCL 2.7e-3 --choiceCLparams 0 --seed  $seed --head 3

