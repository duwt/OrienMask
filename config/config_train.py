from .base import *

orienmask_yolo_coco_544_anchor4_fpn_plus = dict(
    name="OrienMaskAnchor4FPNPlus",
    n_gpu=2,
    epochs=100,
    cudnn_benchmark=True,
    accumulate=1,
    monitor="segm_AP",
    monitor_mode="max",
    log_dir="checkpoints",
    val_freq=5,
    save_freq=20,
    log_freq=50,
    seed=0,
    trainer='Trainer',
    model=orienmask_yolo_fpn_plus_coco,
    train_loader=coco_544_train_loader,
    val_loader=coco_544_val_loader,
    val_gt_file=coco_val2017_gt_file,
    loss=orienmask_yolo_coco_544_anchor4_loss,
    postprocess=orienmask_yolo_coco_544_anchor4_postprocess,
    optimizer=base_sgd,
    lr_scheduler=step_lr_warmup_coco_e100
)


orienmask_yolo_coco_544_anchor4 = construct_config(
    orienmask_yolo_coco_544_anchor4_fpn_plus,
    update=dict(
        name="OrienMaskAnchor4",
        model=orienmask_yolo_coco
    )
)

orienmask_yolo_coco_544 = construct_config(
    orienmask_yolo_coco_544_anchor4,
    update=dict(
        name="OrienMaskBase",
        loss=orienmask_yolo_coco_544_loss,
        postprocess=orienmask_yolo_coco_544_postprocess
    )
)
