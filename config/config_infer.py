from .base import *
from .config_train import *


orienmask_yolo_coco_544_anchor4_fpn_plus_infer = dict(
    n_gpu=1,
    cudnn_benchmark=True,
    model=orienmask_yolo_coco_544_anchor4_fpn_plus['model'],
    transform=transform_infer_544,
    postprocess=orienmask_yolo_coco_544_anchor4_fpn_plus['postprocess'],
    visualizer=coco_visualizer
)


orienmask_yolo_coco_544_anchor4_infer = construct_config(
    orienmask_yolo_coco_544_anchor4_fpn_plus_infer,
    update=dict(model=orienmask_yolo_coco_544_anchor4['model'])
)


orienmask_yolo_coco_544_infer = construct_config(
    orienmask_yolo_coco_544_anchor4_infer,
    update=dict(postprocess=orienmask_yolo_coco_544['postprocess'])
)
