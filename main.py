import os
import argparse
from solver import Solver
from torch.backends import cudnn


def main(config):
    # find optimal set of algorithm on our configuration setting
    cudnn.benchmark = True
    exp_root = config.exp

    log_root = os.path.join(exp_root, config.log_dir)
    model_root = os.path.join(exp_root, config.model_dir)
    sample_root = os.path.join(exp_root, config.sample_dir)
    result_root = os.path.join(exp_root, config.result_dir)

    if not os.path.isdir(exp_root):
        os.makedirs(exp_root, exist_ok=True)
    if not os.path.isdir(log_root):
        os.makedirs(log_root, exist_ok=True)
    if not os.path.isdir(model_root):
        os.makedirs(model_root, exist_ok=True)
    if not os.path.isdir(sample_root):
        os.makedirs(sample_root, exist_ok=True)
    if not os.path.isdir(result_root):
        os.makedirs(result_root, exist_ok=True)

    solver = Solver(config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Config - Model
    parser.add_argument('--z_dim', type=int, defalut=128, help='dimension of random vector')
    parser.add_argument('--w_dim', type=int, defalut=128, help='dimension of w vector')
    parser.add_argument('--n_mapping', type=int, defalut=8, help='the number of mapping network layers')


    # Config - Training
    parser.add_argument()

    # Config - Test


    # Config - Path
    parser.add_argument('exp', type=str, default='StyleGAN_256')
    parser.add_argument('log_dir', type=str, defalut='log')
    parser.add_argument('model_dir', type=str, defalut='model')
    parser.add_argument('sample_dir', type=str, defalut='sample')
    parser.add_argument('result_dir', type=str, defalut='result')

    # Config - Miscellanceous

    config = parser.parse_args()

    print(config)
    main(config)
