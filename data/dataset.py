import json
import os

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """A base dataset built on `torch.utils.data.Dataset`

    Each line of `list_file` contains an image file path, by which
    `_load_sample_data` loads data as a preparation for further
    transformation.
    """

    def __init__(self, list_file, image_dir, anno_file, transform):
        self.samples = pd.read_csv(list_file, header=None)
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = list(self.samples.iloc[idx])
        sample = self._load_sample_data(sample_path)
        sample = self._transform(sample)
        return sample

    def _load_sample_data(self, sample_path):
        raise NotImplementedError

    def _transform(self, sample):
        return self.transform(sample)


class COCODataset(BaseDataset):
    CAT2LABEL = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
        'boat', 'traffic-light', 'fire-hydrant', 'stop-sign', 'parking-meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports-ball', 'kite', 'baseball-bat', 'baseball-glove', 'skateboard',
        'surfboard', 'tennis-racket', 'bottle', 'wine-glass', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot-dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted-plant', 'bed',
        'dining-table', 'toilet', 'tv-monitor', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell-phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy-bear', 'hair-drier', 'toothbrush'
    ]

    def __init__(self, list_file, image_dir, anno_file, transform, with_mask=True, with_info=True):
        super(COCODataset, self).__init__(list_file, image_dir, anno_file, transform)
        self.annotations = json.load(open(self.anno_file))
        self.with_mask = with_mask
        self.with_info = with_info

    def _load_sample_data(self, sample_path):
        image_file = os.path.join(self.image_dir, sample_path[0])
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.float32)
        height, width = image.shape[:2]
        anno = self.annotations[sample_path[0]]['anno']
        bbox = np.array(anno['bbox']).astype(np.float32).reshape(-1, 4)
        cls = np.array(anno['cls']).astype(np.int64)
        sample = {'image': image, 'bbox': bbox, 'cls': cls}

        if self.with_mask:
            mask = [self._convert_mask(anno, height, width) for anno in anno['mask']]
            sample.update({'mask': mask})

        if self.with_info:
            image_id = self.annotations[sample_path[0]]['image_id']
            sample.update({'info': {'id': image_id, 'height': height, 'width': width}})

        return sample

    @staticmethod
    def _convert_mask(anno, height, width):
        if type(anno) == list:
            rles = maskUtils.frPyObjects(anno, height, width)
            rle = maskUtils.merge(rles)
        elif type(anno['counts']) == list:
            rle = maskUtils.frPyObjects(anno, height, width)
        else:
            rle = anno

        mask = maskUtils.decode(rle)
        return mask


class VOCDataset(COCODataset):
    CAT2LABEL = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ]

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'dining-table', 'dog', 'horse', 'motorbike', 'person', 'potted-plant',
        'sheep', 'sofa', 'train', 'tv-monitor'
    ]

    def __init__(self, list_file, image_dir, anno_file, transform, with_mask=False, with_info=True):
        super(VOCDataset, self).__init__(list_file, image_dir, anno_file, transform, with_mask, with_info)
