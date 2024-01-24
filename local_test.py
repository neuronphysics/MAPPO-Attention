from onpolicy.scripts import train
from onpolicy.scripts.train import train_meltingpot
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Melting Pot")

    parser.add_argument('--substrate_name', type=str, default='territory__rooms',
                                help="a physical environment which is paired with different scenarios")

    parser.add_argument('--scenario_name', type=str,
                                default='territory__rooms__0',
                                help="Which scenario to run on [SC 0: killed chef, SC 1: semi-skilled apprentice chef, SC 2:an unhelpful partner ]")

    parser.add_argument('--num_agents', type=int,
                                default=9, help="number of players")

    parser.add_argument('--scale-factor', type=int, default=1, help="the scale factor for the observation")

    parser.add_argument('--use-valuenorm', type=bool, default=True, help="whether to use")


    parser.add_argument('--algorithm-name',type=str,default="mappo")

    parser.add_argument('--cuda',type=bool,default=True)

    parser.add_argument('--env-name',type=str,default="Meltingpot")

    parser.add_argument('--experiment-name',type=str,default="test")





    parser.add_argument('--use-eval', type=bool, default=False)







    parser.add_argument('--num-landmarks', type=int, default=2)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--use-render', type=bool, default=False)

    parser.add_argument('--n-training-threads', type=int, default=1)

    parser.add_argument('--n-rollout-threads', type=int, default=5)

    parser.add_argument('--num-mini-batch', type=int, default=1)

    parser.add_argument('--episode-length', type=int, default=10)

    parser.add_argument('--num-env-steps', type=int, default=200)

    parser.add_argument('--ppo-epoch', type=int, default=15)



    parser.add_argument('--use-wandb', type=bool, default=True)

    parser.add_argument('--user-name', type=str, default='dane-malenfant')

    parser.add_argument('--wandb-name', type=str, default='dane-malenfant')

    parser.add_argument('--share-policy', type=bool, default=False)

    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")

    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")

    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")

    parser.add_argument("--num_env_steps", type=int, default=10,
                        help='Number of environment steps to train (default: 10e6)')


    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--use_attention", type=bool,
                        default=True, help='Whether agent use the attention module or not')
    parser.add_argument("--attention_module", type=str,
                        default='RIM', help='specify the name of attention module')
    parser.add_argument("--use_version_scoff", type=int, default=0, help="specify the version of SCOFF")
    parser.add_argument("--scoff_num_units", type=int, default=4, help="specify the number of units in SCOFF")
    parser.add_argument("--scoff_topk", type=int, default=2, help="specify the number of topk in SCOFF")
    parser.add_argument("----rim_num_units", type=int, default=6, help="specify the number of units in RIM")
    parser.add_argument("--rim_topk", type=int, default=4, help="specify the number of topk in RIM")


    parser.add_argument("--use_centralized_V", action='store_false',
                        default=False, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=96,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")

    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=20,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1,
                        help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")

    # meltingpot parameter
    parser.add_argument("--downsample", action='store_false', default=True,
                        help="the scale factor of each rendered image in saved video.")

    parser.add_argument("--use-recon-loss",  default=True,
                        help="Reconstruction loss for Actor")

    parser.add_argument("--use_kl_loss", default=True,
                        help="kl divergence loss for Actor")

    train_meltingpot.main(parser)

