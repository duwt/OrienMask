from .config_train import *


orienmask_yolo_coco_544_anchor4_fpn_plus_test = dict(
    n_gpu=1,
    cudnn_benchmark=True,
    tester="Tester",
    model=orienmask_yolo_coco_544_anchor4_fpn_plus['model'],
    test_loader=construct_config(
        orienmask_yolo_coco_544_anchor4_fpn_plus['val_loader'],
        update=dict(batch_size=16)
    ),
    postprocess=orienmask_yolo_coco_544_anchor4_fpn_plus['postprocess'],
    gt_file=orienmask_yolo_coco_544_anchor4_fpn_plus['val_gt_file']
)


orienmask_yolo_coco_544_anchor4_test = construct_config(
    orienmask_yolo_coco_544_anchor4_fpn_plus_test,
    update=dict(model=orienmask_yolo_coco_544_anchor4['model'])
)


orienmask_yolo_coco_544_test = construct_config(
    orienmask_yolo_coco_544_anchor4_test,
    update=dict(postprocess=orienmask_yolo_coco_544['postprocess'])
)
