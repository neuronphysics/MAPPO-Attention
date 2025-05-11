#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.meltingpot.MeltingPot_Env import env_creator
from meltingpot import substrate
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import gc
import time
import cProfile
import json
torch.backends.cudnn.benchmark = True
"""Train script for Meltingpot."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Meltingpot":

                player_roles = substrate.get_config(all_args.substrate_name).default_player_roles
                if all_args.downsample:
                    rgb_scale_factor = all_args.img_scale_factor
                    world_scale_factor = all_args.world_img_scale_factor
                else:
                    rgb_scale_factor = 1
                    world_scale_factor = 1
                env_config = {"substrate": all_args.substrate_name, "roles": player_roles, "agent_scale": rgb_scale_factor,
                              "world_scale": world_scale_factor}
                env = env_creator(env_config)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.reset(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Meltingpot":
                player_roles = substrate.get_config(all_args.substrate_name).default_player_roles
                if all_args.downsample:
                    rgb_scale_factor = all_args.img_scale_factor
                    world_scale_factor = all_args.world_img_scale_factor
                else:
                    rgb_scale_factor = 1
                    world_scale_factor = 1
                env_config = {"substrate": all_args.substrate_name, "roles": player_roles,
                              "agent_scale": rgb_scale_factor,
                              "world_scale": world_scale_factor}    
                           
                env = env_creator(env_config)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.reset(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--substrate_name', type=str, default='collaborative_cooking',
                        help="a physical environment which is paired with different scenarios")
    parser.add_argument('--scenario_name', type=str,
                        default='collaborative_cooking__circuit_0',
                        help="Which scenario to run on [SC 0: killed chef, SC 1: semi-skilled apprentice chef, SC 2:an unhelpful partner ]")
    parser.add_argument("--roles", type=str, default='default')

    parser.add_argument("--num_agents", type=int, default=16,
                        help="number of controlled players.")

    parser.add_argument('--scale_factor', type=int, default=1, help="the scale factor for the observation")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def save_config_to_json(config, filename):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Modify filename to avoid overwriting
    base_filename, file_extension = os.path.splitext(filename)
    counter = 1
    unique_filename = base_filename
    while os.path.exists('logs/' + unique_filename + file_extension):
        unique_filename = f"{base_filename}_{counter}"
        counter += 1

    # Save the config to the file
    with open('logs/' + unique_filename + file_extension, 'w') as file:
        json.dump(vars(config), file, indent=4)


def main(args):
    torch.cuda.empty_cache()
    gc.collect()
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(f"TRAIN, value of use_attention is {all_args.use_attention}")
        if all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy:
            print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be True")
        if all_args.use_attention and not (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy):
            print(
                "u are choosing to use mappo, we set use attention to be True and use_recurrent_policy & use_naive_recurrent_policy to be False")
            all_args.use_recurrent_policy = False
            all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        print("u are choosing to use mat, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    print(all_args)
    if all_args.substrate_name == 'collaborative_cooking':
        substrate.get_config(all_args.substrate_name).cooking_pot_pseudoreward = 1.0

        # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    base_run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.substrate_name / all_args.algorithm_name / all_args.experiment_name
    

    # Create specific run directory (both for wandb and non-wandb cases)
    # Create unique run directory regardless of wandb usage
    if not base_run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in base_run_dir.iterdir() 
                        if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
            
    run_dir = base_run_dir / curr_run
    all_args.log_dir = run_dir  # Set log_dir regardless of wandb usage
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        
        if all_args.use_sweep_wandb_hyper_search:
            all_args.lr = wandb.config.lr
            all_args.critic_lr = wandb.config.critic_lr
            all_args.entropy_coef = wandb.config.entropy_coef
            all_args.entropy_final_coef = wandb.config.entropy_final_coef
            all_args.clip_param = wandb.config.clip_param
            all_args.max_grad_norm = wandb.config.max_grad_norm
            all_args.gain = wandb.config.gain
            

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.meltingpot_runner import MeltingpotRunner as Runner
    else:
        from onpolicy.runner.separated.meltingpot_runner import MeltingpotRunner as Runner

    #################
    # debug with vs code
    # ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)

    # Pause the program until a remote debugger is attached
    # ptvsd.wait_for_attach()

    ###################
    runner = Runner(config)
    # cProfile.runctx('runner.run()', globals(), locals(), 'profile_run.prof')
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
