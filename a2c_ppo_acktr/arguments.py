import argparse
import torch
from os.path import join

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr', choices=['a2c', 'ppo', 'acktr'])
    parser.add_argument('--device', type=str, default='cuda:4', help='gpu device')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer alpha')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--use_gae', action='store_true', default=True, help='use generalized advantage estimation')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='gae lambda parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=1e-5, help='value loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda_deterministic',action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num_steps', type=int, default=128, help='number of steps in an episode or number of forward steps in A2C')
    parser.add_argument('--ppo_epoch', type=int, default=100, help='number of ppo epochs')
    parser.add_argument('--num_mini_batch', type=int, default=1, help='number of batches for ppo')
    parser.add_argument('--clip_param', type=float, default=0.2, help='ppo clip parameter')
    parser.add_argument('--log_interval', type=int, default=1e2, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=1e4, help='save interval, one save per n updates')
    parser.add_argument('--eval_interval', type=int, default=1, help='eval interval, one eval per n updates')
    parser.add_argument('--num_env_steps', type=int, default=1e6, help='number of environment steps to train')
    parser.add_argument('--output_dir', type=str, default='result-GAE-2', help='save all outputs here')
    parser.add_argument('--recurrent_policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=True, help='use a linear schedule on the learning rate')

    # I think these will not be used
    parser.add_argument('--num_processes', type=int, default=1, help='how many training CPU processes to use')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False, help='compute returns taking into account time limits')

    args = parser.parse_args()

    args.log_dir = join(args.output_dir, 'logs')
    args.save_dir = join(args.output_dir, 'checkpoints')

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

    

    return args
