import copy


MEAN = [123.675, 116.280, 103.530]
STD = [58.395, 57.120, 57.375]
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
ANCHORS_YOLOV3 = [
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
]
ANCHORS_YOLOV4 = [
    [12, 16], [19, 36], [40, 28],
    [36, 75], [76, 55], [72, 146],
    [142, 110], [192, 243], [459, 401]
]


def construct_config(config, update=None, pop=None):
    """Construct config from a base config

    If a key of `update` matches `config` and both of their values are `dict`,
    then the update process will be iteratively executed. Otherwise, it is the
    same as built-in `update` function of `dict` type.

    The items in `pop` are iterative keys joined by periods. For example,
    `top_key.sub_key' means popping config['top_key']['sub_key']. If there is no
    period included, then the pop process will be the same as built-in `pop`
    function of `dict` type.

    Args:
        config (dict): the base config
        update (dict): update on base config
        pop (list): pop keys out of base config
    """
    new_config = copy.deepcopy(config)
    if update is not None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(new_config.get(key), dict):
                new_config[key] = construct_config(new_config[key], update=value)
            else:
                new_config[key] = value
    if pop is not None:
        for key in pop:
            sub_keys = key.split('.')
            sub_config = new_config
            for sub_key in sub_keys[:-1]:
                sub_config = sub_config[sub_key]
            sub_config.pop(sub_keys[-1])
    return new_config


# train configuration template
template_train = dict(
    name=None,              # used to create checkpoint sub-folder
    n_gpu=None,             # number of gpu devices to train
    epochs=None,            # total epochs over the train dataset
    cudnn_benchmark=None,   # to speed up convolution if the input size is fixed
    accumulate=None,        # accumulate gradients over multiple mini-batches
    monitor=None,           # the criterion for saving the best model
    monitor_mode=None,      # three options: 'min', 'max', 'off'
    log_dir=None,           # checkpoints base directory
    val_freq=None,          # validation interval (epochs)
    save_freq=None,         # interval (epochs) to save training checkpoints
    log_freq=None,          # logging interval (batches) on tensorboard
    seed=None,              # random seed
    trainer=None,           # trainer type
    model=None,
    train_loader=None,
    val_loader=None,
    val_gt_file=None,
    postprocess=None,
    loss=None,
    optimizer=None,
    lr_scheduler=None
)

template_test = dict(
    n_gpu=None,
    cudnn_benchmark=None,
    tester=None,
    model=None,
    test_loader=None,
    postprocess=None,
    gt_file=None
)

template_infer = dict(
    n_gpu=None,
    cudnn_benchmark=True,
    model=None,
    transform=None,
    postprocess=None,
    visualizer=None
)


# model configurations
orienmask_yolo_coco = dict(
    type="OrienMaskYOLO",
    num_anchors=3,
    num_classes=80,
    pretrained="checkpoints/pretrained/pretrained_darknet53.pth",
    freeze_backbone=False,
    backbone_batchnorm_eval=False
)

orienmask_yolo_fpn_plus_coco = construct_config(
    orienmask_yolo_coco,
    update=dict(type="OrienMaskYOLOFPNPlus")
)


# dataset configurations
coco_train_dataset = dict(
    type="COCODataset",
    list_file="coco/list/coco_train.txt",
    image_dir="coco/train2017",
    anno_file="coco/annotations/orienmask_coco_train.json",
    with_mask=True,
    with_info=False
)

coco_val_dataset = dict(
    type="COCODataset",
    list_file="coco/list/coco_val.txt",
    image_dir="coco/val2017",
    anno_file="coco/annotations/orienmask_coco_val.json",
    with_mask=True,
    with_info=True
)


# transform configurations
transform_train_544 = dict(
    type="COCOTransform",
    pipeline=[
        dict(type="ColorJitter", brightness=0.2, contrast=0.5, saturation=0.5, hue=0.1),
        dict(type="RandomCrop", p=0.5, image_min_iou=0.64, bbox_min_iou=0.64),
        dict(type="Resize", size=(544, 544), pad_needed=True, warp_p=0.25, jitter=0.3,
             random_place=True, pad_p=0.75, pad_ratio=0.75, pad_value=MEAN),
        dict(type="RandomHorizontalFlip", p=0.5),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0, 0, 0), std=(255, 255, 255))
    ]
)

transform_val_544 = dict(
    type="COCOTransform",
    pipeline=[
        dict(type="Resize", size=(544, 544), pad_needed=False, warp_p=0., jitter=0.,
             random_place=False, pad_p=0., pad_ratio=0., pad_value=MEAN),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=(0, 0, 0), std=(255, 255, 255))
    ]
)

transform_infer_544 = dict(
    type="FastCOCOTransform",
    pipeline=[
        dict(type="Resize", size=(544, 544), interpolation='bilinear', align_corners=False),
        dict(type="Normalize", mean=(0, 0, 0), std=(255, 255, 255))
    ]
)


# dataloader configurations
coco_544_train_loader = dict(
    type="DataLoader",
    dataset=coco_train_dataset,
    transform=transform_train_544,
    batch_size=8,
    num_workers=2,
    shuffle=True,
    pin_memory=False,
    collate=dict(type="collate")
)

coco_544_val_loader = dict(
    type="DataLoader",
    dataset=coco_val_dataset,
    transform=transform_val_544,
    batch_size=8,
    num_workers=2,
    shuffle=False,
    pin_memory=False,
    collate=dict(type="collate")
)


# ground truth files for coco evaluation
coco_train2017_gt_file = "coco/annotations/instances_train2017.json"
coco_val2017_gt_file = "coco/annotations/instances_val2017.json"


# loss configurations
orienmask_yolo_coco_544_loss = dict(
    type="OrienMaskYOLOMultiScaleLoss",
    grid_size=[[17, 17], [34, 34], [68, 68]],
    image_size=[544, 544],
    anchors=ANCHORS_YOLOV3,
    anchor_mask=ANCHORS_MASK,
    num_classes=80,
    center_region=0.6,
    valid_region=0.6,
    label_smooth=False,
    obj_ignore_threshold=0.7,
    weight=[1, 1, 1, 1, 1, 20, 20],
    scales_weight=[1, 1, 1]
)

orienmask_yolo_coco_544_anchor4_loss = construct_config(
    orienmask_yolo_coco_544_loss,
    update=dict(anchors=ANCHORS_YOLOV4)
)


# postprocess configurations
orienmask_yolo_coco_544_postprocess = dict(
    type="OrienMaskYOLOPostProcess",
    grid_size=[[17, 17], [34, 34], [68, 68]],
    image_size=[544, 544],
    anchors=ANCHORS_YOLOV3,
    anchor_mask=ANCHORS_MASK,
    num_classes=80,
    conf_thresh=0.005,
    nms=dict(type='batched_nms', threshold=0.5),
    nms_pre=400,
    nms_post=100,
    orien_thresh=0.3
)

orienmask_yolo_coco_544_anchor4_postprocess = construct_config(
    orienmask_yolo_coco_544_postprocess,
    update=dict(anchors=ANCHORS_YOLOV4)
)


# optimizer configurations
base_sgd = dict(
    type="SGD",
    lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
)


# learning rate scheduler configurations
step_lr_warmup_coco_e100 = dict(
    type="StepWarmUpLR",
    warmup_type="linear",
    warmup_iter=1000,
    warmup_ratio=0.1,
    milestones=[520000, 660000],
    gamma=0.1
)


# visualizer configurations
coco_visualizer = dict(
    type="InferenceVisualizer",
    dataset="COCO",
    with_mask=True,
    conf_thresh=0.3,
    alpha=0.6,
    line_thickness=1
)
