import argparse

from distutils.util import strtobool


def str2bool(v):
    return bool(strtobool(v))


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='ippo', choices=["ippo", "rmappo", "mappo"])

    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=2, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", type=str2bool, default=True,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        type=str2bool, default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=40e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='zsheikhb',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", type=str2bool, default=False,
                        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--use_sweep_wandb_hyper_search", type=str2bool, default=False,
                        help="hyper parameter  search")
    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", type=str2bool,
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=2000, help="Max length for any episode")

    # network parameters
    parser.add_argument("--use_attention", type=str2bool, default=False,
                        help='Whether agent use the attention module or not')

    parser.add_argument("--attention_module", type=str,
                        default='RIM', help='specify the name of attention module')
    parser.add_argument("--use_version_scoff", type=int, default=1, help="specify the version of SCOFF")
    parser.add_argument("--scoff_num_units", type=int, default=4, help="specify the number of units in SCOFF")
    parser.add_argument("--scoff_topk", type=int, default=3, help="specify the number of topk in SCOFF")
    parser.add_argument("--scoff_do_relational_memory", type=str2bool, default=False,
                        help="specify whether if we use relational memory")

    parser.add_argument("--rim_num_units", type=int, default=6, help="specify the number of units in RIM")
    parser.add_argument("--rim_topk", type=int, default=3, help="specify the number of topk in RIM")

    parser.add_argument("--share_policy", type=str2bool,
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", type=str2bool,
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", type=str2bool,
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=144,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", type=str2bool,
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", type=str2bool, default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", type=str2bool, default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", type=str2bool,
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", type=str2bool, default=False,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.1,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", type=str2bool,
                        default=True, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", type=str2bool,
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        type=str2bool, default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    # entropy_coef params
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--entropy_final_coef", type=float, default=0.01,
                        help='final entropy coefficiennt value after annealing')
    parser.add_argument("--entropy_anneal_duration", type=float, default=1000000,
                        help='duration of entropy annealing')
    parser.add_argument("--warmup_updates", type=int, default=50000,
                        help='number of warmup updates')
    parser.add_argument("--cooldown_updates", type=int, default=50000,
                        help='number of cooldown updates')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        type=str2bool, default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", type=str2bool,
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", type=str2bool,
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", type=str2bool, default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        type=str2bool, default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        type=str2bool, default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=5.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", type=str2bool,
                        default=True, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", type=str2bool, default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", type=str2bool, default=True,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", type=str2bool, default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=1, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1,
                        help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")

    # meltingpot parameter
    parser.add_argument("--downsample", type=str2bool, default=True,
                        help="the scale factor of each rendered image in saved video.")
    parser.add_argument("--img_scale_factor", type=int, default=8,
                        help="the scale factor of each rendered image in saved video.")

    parser.add_argument("--world_img_scale_factor", type=int, default=8,
                        help="the scale factor of each rendered image in saved video.")

    # additional modular attention parameters
    parser.add_argument("--drop_out", type=float,
                        default=0.5, help="specify the drop out")
    parser.add_argument("--rnn_attention_module", type=str,
                        default='GRU', help='specify the rnn module to use')
    parser.add_argument("--use_bidirectional", type=str2bool, default=False,
                        help='Whether or not to be bidirectional')

    parser.add_argument("--load_model", type=str2bool, default=False,
                        help='Whether or not we load the pretrained model')

    parser.add_argument("--use_pos_encoding", type=str2bool, default=False, )
    parser.add_argument("--use_input_att", type=str2bool, default=False, )
    parser.add_argument("--use_com_att", type=str2bool, default=False, )
    parser.add_argument("--use_x_reshape", type=str2bool, default=False, )

    parser.add_argument("--scoff_num_modules_read_input", type=int, default=4, )
    parser.add_argument("--scoff_inp_heads", type=int, default=4, )
    parser.add_argument("--scoff_share_comm", type=str2bool, default=True, )
    parser.add_argument("--scoff_share_inp", type=str2bool, default=True, )
    parser.add_argument("--scoff_memory_mlp", type=int, default=4, )
    parser.add_argument("--scoff_memory_slots", type=int, default=4, )
    parser.add_argument("--scoff_memory_head_size", type=int, default=32, )
    parser.add_argument("--scoff_num_memory_heads", type=int, default=4, )
    parser.add_argument("--scoff_memory_topk", type=int, default=4, )
    parser.add_argument("--scoff_num_schemas", type=int, default=3, help="number of schemas")

    parser.add_argument("--pretrain_slot_att", type=str2bool, default=False, )
    
    parser.add_argument("--crop_size", type=int, default=44, )
    parser.add_argument("--slot_train_ep", type=int, default=2000, )
    parser.add_argument('--slot_att_crop_repeat', type=int, default=5)
    parser.add_argument("--slot_clip_grade_norm", type=float, default=1.0)
    parser.add_argument("--slot_save_fre", type=int, default=5, )
    parser.add_argument("--slot_log_fre", type=int, default=3, )

    parser.add_argument("--use_consistency_loss", type=str2bool, default=True)
    parser.add_argument("--use_orthogonal_loss", type=str2bool, default=False, )

    parser.add_argument("--slot_att_work_path", type=str,
                        default="/mnt/e/pycharm_projects/meltingpot-main/onpolicy/scripts/results/slot_att/", )
    parser.add_argument("--slot_pretrain_batch_size", type=int, default=2, )

    parser.add_argument("--slot_att_load_model", type=str2bool, default=False, )
    parser.add_argument("--use_slot_att", type=str2bool, default=True, )
    parser.add_argument("--collect_data_ep_num", type=int, default=10, )
    parser.add_argument("--collect_data", type=str2bool, default=False, )
    parser.add_argument("--collect_agent", type=str2bool, default=False, )
    parser.add_argument("--collect_world", type=str2bool, default=False, )
    parser.add_argument("--no_train", type=str2bool, default=False, )

    parser.add_argument('--grad_clip', type=float, default=1.0)
 
    parser.add_argument('--drop_path', type=float, default=0.2)
    parser.add_argument('--dvae_kernel_size', type=int, default=3)
    parser.add_argument('--truncate', type=str, default='bi-level', help='bi-level or fixed-point or none')

    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--decay_steps', type=int, default=50000)

    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--init_size', type=int, default=256)
    parser.add_argument('--mlp_size', type=int, default=256)

    parser.add_argument('--feature_size', type=int, default=128)
    parser.add_argument('--encoder_channels', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--encoder_strides', type=int, nargs='+', default=[1, 1, 1, 1])
    parser.add_argument('--encoder_kernel_size', type=int, default=5)
    parser.add_argument('--img_channels', type=int, default=3)

    parser.add_argument('--init_method', default='embedding', help='embedding or shared_gaussian')

    parser.add_argument('--tau_steps', type=int, default=30000)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_start', type=float, default=1)

    parser.add_argument('--sigma_steps', type=int, default=30000)
    parser.add_argument('--sigma_final', type=float, default=0)
    parser.add_argument('--sigma_start', type=float, default=1)

    parser.add_argument('--use_post_cluster', default=False, action='store_true')
    parser.add_argument('--use_kmeans', default=False, action='store_true')
    parser.add_argument('--lambda_c', type=float, default=0.1)
    parser.add_argument('--lr_main', type=float, default=1e-4)
    parser.add_argument('--lr_dvae', type=float, default=3e-4)

    parser.add_argument('--clip_beta', type=float, default= 2.0, help='Weight Clipping beta value')
    parser.add_argument('--slot_attn_loss_coef', type=float, default= 0.05, help='Slot Attention Loss Coefficient')
    parser.add_argument('--collect_data_mi', type=int, default=50000)
    parser.add_argument('--fine_tuning_type', type=str, default="Lora")
    

    return parser
