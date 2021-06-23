import argparse
import json

import torch
import torch.distributed as dist

import config as config_module
from trainer.builder import build_trainer


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='multi-gpu rank provided by torch.distributed.launch')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='checkpoint to resume training (default: None)')
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='weights to start training (default: None)')
    args = parser.parse_args()

    # Load config file
    if args.config is not None:
        if args.config.endswith('.json'):
            train_config = json.load(open(args.config))
        else:
            train_config = getattr(config_module, args.config)
    elif args.resume is not None:
        train_config = torch.load(args.resume, map_location='cpu')['config']
    else:
        raise AssertionError("Configuration file need to be specified.")

    # multi-gpu settings
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = train_config.get('cudnn_benchmark', True)
    assert torch.cuda.device_count() == train_config['n_gpu']
    if train_config['n_gpu'] > 1:
        device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group('nccl')

    # build trainer and train
    trainer = build_trainer(train_config, device, args)
    trainer.train()
