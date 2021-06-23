import copy
import os
import random
import functools

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import data as data_module
import eval as evaluation_module
import eval.function as eval_func_module
import model as model_module
import optim as optim_module
from utils.envs import get_world_size
import trainer.tester as tester_module
import trainer.trainer as trainer_module


def build_trainer(config, device, args):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    is_distributed = get_world_size() > 1
    train_loader = build_dataloader(config['train_loader'], is_distributed)
    val_loader = build_dataloader(config['val_loader'], is_distributed)
    postprocess = build_postprocess(config['postprocess'], device=device)
    ignore_pretrained = (args.resume or args.weights)
    model = build_model(config['model'], is_distributed, ignore_pretrained)
    loss = build(config['loss'], evaluation_module)
    optimizer = build_optimizer(config['optimizer'], config['accumulate'], model, is_distributed)
    lr_scheduler = build(config['lr_scheduler'], optim_module, optimizer=optimizer)
    trainer_type = getattr(trainer_module, config.get('trainer', 'Trainer'))
    trainer = trainer_type(model, loss, optimizer, lr_scheduler, config,
                           train_loader, val_loader, postprocess, device, args)
    return trainer


def build_tester(config, checkpoint):
    test_config = copy.deepcopy(config)
    train_config = torch.load(checkpoint)['config']
    device = torch.device('cuda:0' if test_config['n_gpu'] > 0 else 'cpu')
    test_loader = build_dataloader(test_config['test_loader'])
    model_config = copy.deepcopy(train_config)['model']
    model_config['pretrained'] = None
    model = build(model_config, model_module).to(device)
    weights = torch.load(checkpoint, map_location=device)['state_dict']
    model.load_state_dict(weights, strict=True)
    postprocess = build_postprocess(config['postprocess'], device=device)
    checkpoint_dir = os.path.dirname(checkpoint)
    tester_type = getattr(tester_module, config.get('tester', 'Tester'))
    tester = tester_type(model, postprocess, test_loader,
                         checkpoint_dir, device, test_config['gt_file'])
    return tester


def build(config, module, **kwargs):
    build_config = copy.deepcopy(config)
    class_name = build_config.pop('type')
    return getattr(module, class_name)(**build_config, **kwargs)


def build_func_partial(config, module, **kwargs):
    build_config = copy.deepcopy(config)
    func_name = build_config.pop('type')
    return functools.partial(getattr(module, func_name), **build_config, **kwargs)


def build_postprocess(config, device):
    postprocess_config = copy.deepcopy(config)
    nms_config = postprocess_config.pop('nms', None)
    nms_func = build_func_partial(nms_config, eval_func_module) if nms_config else None
    return build(postprocess_config, evaluation_module, nms_func=nms_func, device=device)


def build_model(config, is_distributed=False, ignore_pretrained=False):
    model_config = copy.deepcopy(config)
    if ignore_pretrained:
        model_config['pretrained'] = None
    model = build(model_config, model_module).cuda()
    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[dist.get_rank()])
    return model


def build_dataloader(config, is_distributed=False):
    dataloader_config = copy.deepcopy(config)
    dataset_config = dataloader_config.pop('dataset')
    transform_config = dataloader_config.pop('transform')
    transform = build_transform(transform_config)
    dataset_config['transform'] = transform
    dataset = build(dataset_config, data_module)
    dataloader_config['dataset'] = dataset
    collate_config = dataloader_config.pop('collate', {'type': 'collate'})
    collate_fn = build_func_partial(collate_config, data_module)
    dataloader_config['collate_fn'] = collate_fn
    if is_distributed:
        shuffle = dataloader_config.pop('shuffle', False)
        dataloader_config['sampler'] = DistributedSampler(dataset, shuffle=shuffle)
    return build(dataloader_config, data_module)


def build_transform(config):
    kwargs = copy.deepcopy(config)
    class_name = kwargs.pop('type')
    pipeline_config = kwargs.pop('pipeline')
    transform_class = getattr(data_module, class_name)
    pipeline = [build(item, transform_class) for item in pipeline_config]
    kwargs['pipeline'] = pipeline
    return transform_class(**kwargs)


def build_optimizer(config, accumulate, model, is_distributed=False):
    model = model.module if is_distributed else model
    optimizer_config = copy.deepcopy(config)
    optimizer_config['lr'] = optimizer_config['lr'] / accumulate
    param_groups_config = optimizer_config.pop('param_groups', None)
    if param_groups_config:
        param_groups_config['base_lr'] = optimizer_config['lr']
        param_groups_config['weight_decay'] = optimizer_config['weight_decay']
        trainable_params = getattr(optim_module, 'param_groups')(model, **param_groups_config)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    return build(optimizer_config, optim_module, params=trainable_params)
