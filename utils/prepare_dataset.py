import os
import sys
import json
import random


class COCOConverter:
    """
    refer to http://cocodataset.org/#format-data for more detailed information

    COCO annotation json file format (val)
        {
            'info': {'description', 'url', 'version'=, 'year'=, 'contributor'=, 'date_created'},
            'licenses': [ {'url', 'id', 'name'} * 8 ],
            'images': [ {'license', 'file_name', 'coco_url', 'height',
                         'width', 'date_captured', 'flickr_url', 'id'} * 5000 ],
            'annotations': [ {'segmentation', 'area', 'iscrowd', 'image_id',
                              'bbox', 'category_id', 'id'} * 36781 ],
            'categories': [ {'supercategory', 'id', 'name'} * 80 ]
        }
    -> label json format
        {
            'file_name1': {
                'anno': { 'bbox', 'cls', 'mask' }
                'image_id':
            },
            'file_name2': {
                'anno': { 'bbox', 'cls', 'mask' }
                'image_id':
            },
            ...
        }

    data folder structure:
        base_dir
        ├── train2017
        ├── val2017
        ├── annotations
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        │   ├── coco_train.json (new)
        │   ├── coco_val.json (new)
        ├── list (new)
        │   ├── coco_train.txt (new)
        │   ├── coco_val.txt (new)
    """

    def __init__(self, image_dir, anno_file, label_file, list_file, with_mask=True):
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.label_file = label_file
        self.list_file = list_file
        self.with_mask = with_mask
        os.makedirs(os.path.dirname(self.list_file), exist_ok=True)
        self._create_index()

    def create_dataset_list(self, seed=0):
        image_names = list(self.generate_bbox_mask())
        random.seed(seed)
        random.shuffle(image_names)
        with open(self.list_file, 'w') as handle:
            handle.write('\n'.join(image_names))

    def generate_bbox_mask(self):
        anno_dict = dict()
        for image_id, image_idx in self.img2idx.items():
            image_info = self.coco['images'][image_idx]
            height, width = image_info['height'], image_info['width']

            bboxes, categories, masks = [], [], []
            for anno_idx in self.img2anno[image_id]:
                anno_info = self.coco['annotations'][anno_idx]
                if anno_info['iscrowd'] or anno_info['area'] < 1:
                    continue
                anno = self._anno_to_label(anno_info, height, width)
                bbox, category = anno[0], anno[1]
                if bbox[2] < 1e-8 or bbox[3] < 1e-8:
                    continue
                bboxes.append(bbox)
                categories.append(category)
                if self.with_mask:
                    masks.append(anno[2])

            # if len(bboxes) == 0:
            #     print(image_info['file_name'])
            #     continue

            file_name = image_info['file_name']
            anno_dict[file_name] = {'anno': {'bbox': bboxes, 'cls': categories},
                                    'image_id': image_id}
            if self.with_mask:
                anno_dict[file_name]['anno']['mask'] = masks

            print("\rProgress: {0}".format(len(anno_dict)), end=' ')
            sys.stdout.flush()

        print('\nDone')

        with open(self.label_file, 'w') as handle:
            json.dump(anno_dict, handle)
        return anno_dict.keys()

    def _create_index(self):
        self.coco = json.load(open(self.anno_file))
        self.cat2label = {cat_info['id']: i for i, cat_info in enumerate(self.coco['categories'])}
        self.img2idx = {image_info['id']: i for i, image_info in enumerate(self.coco['images'])}
        self.img2anno = {image_id: [] for image_id in self.img2idx.keys()}
        for i, anno_info in enumerate(self.coco['annotations']):
            self.img2anno[anno_info['image_id']].append(i)

    def _anno_to_label(self, anno_info, height, width):
        # coco [x_left_top, y_left_top, w, h] pixels
        # -> [x_center, y_center, w, h] normalized
        anno_bbox = anno_info['bbox']
        bbox = [0, 0, 0, 0]
        bbox[0] = (anno_bbox[0] + anno_bbox[2] / 2) / width
        bbox[1] = (anno_bbox[1] + anno_bbox[3] / 2) / height
        bbox[2] = anno_bbox[2] / width
        bbox[3] = anno_bbox[3] / height
        category = self.cat2label[anno_info['category_id']]
        if self.with_mask:
            mask = anno_info['segmentation']
            return bbox, category, mask
        else:
            return bbox, category


if __name__ == '__main__':
    base_dir = 'coco'
    for data_type in ('val', 'train'):
        image_dir = os.path.join(base_dir, '{0}2017'.format(data_type))
        anno_file = os.path.join(base_dir, 'annotations/instances_{}2017.json'.format(data_type))
        label_file = os.path.join(base_dir, 'annotations/orienmask_coco_{}.json'.format(data_type))
        list_file = os.path.join(base_dir, 'list/coco_{}.txt'.format(data_type))
        coco = COCOConverter(image_dir, anno_file, label_file, list_file, with_mask=True)
        coco.create_dataset_list(seed=3)
