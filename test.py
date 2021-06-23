import argparse
import json

import torch

import config as config_module
from trainer.builder import build_tester

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file or module (default: None)')
    parser.add_argument('-w', '--checkpoint', default=None, type=str,
                        help='model checkpoint to test (default: None)')
    args = parser.parse_args()

    if args.config.endswith('.json'):
        test_config = json.load(open(args.config))
    else:
        test_config = getattr(config_module, args.config)

    assert test_config['n_gpu'] <= 1, 'Multi-gpu test is not supported'
    if test_config['n_gpu'] > 0:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = test_config.get('cudnn_benchmark', True)

    tester = build_tester(test_config, args.checkpoint)
    tester.test()
