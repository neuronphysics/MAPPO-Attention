#!/bin/bash
#./train_meltingpot.sh TERRITORY 0 RIM 100 6 3 SAF
environment=$1
seed=$2
module=$3
hidden=$4
units=$5
topk=$6
skill=$7

# Default values in case the environment does not match
agents=0
substrate=""
episode_length=0
bottom=0
sup=0

if [ "$skill" = "SAF" ]; then
    #conda shell.bash activate saf-melt
    #wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
    #conda activate saf-melt
    echo "SAF"
else
    conda init bash
    conda shell.bash activate marl
    #wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
    conda activate marl
fi

# Set agents and substrate name based on the environment
case "$environment" in
    "HARVEST")
        echo "RUNNING HARVEST"
        agents=16
        substrate="allelopathic_harvest__open"
        episode_length=2000
        bottom=32
        sup=24
        ;;
    "TERRITORY")
        echo "RUNNING territory__rooms"
        agents=9
        substrate="territory__rooms"
        episode_length=1000
        bottom=18
        sup=14
        ;;
    "CLEAN")
        echo "RUNNING clean_up"
        agents=7
        substrate="clean_up"
        episode_length=1000
        bottom=14
        sup=11
        ;;
    "CHEMISTRY")
        echo "RUNNING chemistry"
        agents=8
        substrate="chemistry__three_metabolic_cycles_with_plentiful_distractors"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    "STRAVINSKY")
        echo "RUNNING bach or stravinsky"
        agents=8
        substrate="bach_or_stravinsky_in_the_matrix__arena"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    "COOKING")
        echo "RUNNING coolaborative cooking"
        agents=2
        substrate="collaborative_cooking__cramped"
        episode_length=1000
        bottom=4
        sup=3
        ;;
    "PRISONERS")
        echo "RUNNING prisoners_dilemma_in_the_matrix__arena"
        agents=8
        substrate="prisoners_dilemma_in_the_matrix__arena"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    *)
        echo "Unknown environment: $environment"
        exit 1
        ;;
esac

echo "Environment: $environment"
echo "Seed: $seed"
echo "Module: $module"
echo "Hidden: $hidden"
echo "Units: $units"
echo "using: $skills"


if [ "$skill" = "SKILLS" ]; then
    export PYTHONPATH="${PYTHONPATH}:/home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_skills/MAPPO-ATTENTIOAN-Intrinsic_rewards"

    echo "Using skill"
    # Execute the program based on the module
    if [ "$module" = "RIM" ]; then
        echo "Executing program for RIM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_skills/MAPPO-ATTENTIOAN-Intrinsic_rewards/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --rim_num_units ${units} --rim_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    elif [ "$module" = "SCOFF" ]; then
        echo "Executing program for SCOFF"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_skills/MAPPO-ATTENTIOAN-Intrinsic_rewards/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --scoff_num_units ${units} --scoff_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    elif [ "$module" = "LSTM" ]; then
        echo "Executing program for LSTM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_skills/MAPPO-ATTENTIOAN-Intrinsic_rewards/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention False --use_naive_recurrent_policy True --use_recurrent_policy True --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    else
        echo "Module is neither RIM nor SCOFF, nor LSTM"
    fi


elif [ "$skill" = "SAF_O" ] || [ "$skill" = "SAF_M" ]; then
    export PYTHONPATH="${PYTHONPATH}:/home/juan-david-vargas-mazuera/ICML-RUNS/CODES/saf"
    echo "Executing SAF"

    if [ "$skill" = "SAF_O" ]; then
        echo "Compound goal env"
        env=CompoundGoalEnv
    else
        echo "Meltingpot env"
        env=meltingpot
    fi
    
    N_agents=9
    Method=ippo
    coordination=False
    heterogeneity=True
    use_policy_pool=False
    latent_kl=False
    ProjectName=saf_marlgrid

    ExpName=${env}"_"${N_agents}"_"${coordination}"_"${heterogeneity}"_"${Method}"-"${use_policy_pool}"-"${latent_kl}"_"${seed}

    HYDRA_FULL_ERROR=1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/saf/run.py \
    env=marlgrid  \
    env.name=${env} \
    env.params.max_steps=50 \
    env.params.coordination=${coordination} \
    env.params.heterogeneity=${heterogeneity} \
    seed=${seed} \
    n_agents=${N_agents} \
    env_steps=50 \
    env.params.num_goals=100 \
    experiment_name=${ExpName} \
    policy=${Method} \
    policy.params.type=conv \
    policy.params.activation=tanh \
    policy.params.update_epochs=10 \
    policy.params.num_minibatches=1 \
    policy.params.learning_rate=0.0007 \
    policy.params.clip_vloss=True \
    runner.params.lr_decay=False \
    runner.params.comet.project_name=$ProjectName \
    use_policy_pool=${use_policy_pool} \
    latent_kl=${latent_kl} \

else
    export PYTHONPATH="${PYTHONPATH}:/home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_base/MAPPO-ATTENTIOAN"

    echo "Not using skill"
    # Execute the program based on the module
    if [ "$module" = "RIM" ]; then
        echo "Executing program for RIM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_base/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py --rim_num_units ${units} --rim_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    elif [ "$module" = "SCOFF" ]; then
        echo "Executing program for SCOFF"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_base/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py --scoff_num_units ${units} --scoff_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    elif [ "$module" = "LSTM" ]; then
        echo "Executing program for LSTM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/juan-david-vargas-mazuera/ICML-RUNS/CODES/repo_base/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention False --use_naive_recurrent_policy True --use_recurrent_policy True --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    else
        echo "Module is neither RIM nor SCOFF, nor LSTM"
    fi

fi 