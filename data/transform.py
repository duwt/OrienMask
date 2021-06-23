import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.nn.modules.utils import _pair
from torchvision.transforms.transforms import Lambda, Compose

_cv2_interpolation_to_str = {
    cv2.INTER_NEAREST: 'cv2.INTER_NEAREST',
    cv2.INTER_LINEAR: 'cv2.INTER_LINEAR',
    cv2.INTER_AREA: 'cv2.INTER_AREA',
    cv2.INTER_CUBIC: 'cv2.INTER_CUBIC',
    cv2.INTER_LANCZOS4: 'cv2.INTER_LANCZOS4'
}

_str_to_cv2_interpolation = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4
}


def adjust_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
    return np.clip(image * brightness_factor, 0, 255)


def adjust_contrast(image: np.ndarray, contrast_factor: float) -> np.ndarray:
    mean_value = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()
    return np.clip(image * contrast_factor + mean_value * (1 - contrast_factor), 0, 255)


def adjust_saturation(image: np.ndarray, saturation_factor: float) -> np.ndarray:
    gray_image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=2)
    return np.clip(image * saturation_factor + gray_image * (1 - saturation_factor), 0, 255)


def adjust_hue(image: np.ndarray, hue_factor: float) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[..., 0] = np.clip(hsv_image[..., 0] + hue_factor * 360, 0, 360)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def imresize(image: np.ndarray, size, interpolation) -> np.ndarray:
    return cv2.resize(image, size, interpolation=interpolation)


def impad(image: np.ndarray, padding, value=0.) -> np.ndarray:
    return cv2.copyMakeBorder(image, *padding, cv2.BORDER_CONSTANT, value=value)


def imhflip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1)


def imvflip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=0)


class BaseTransform:
    """A base transform modified from `torchvision.transforms.transforms'

    All transforms are implemented as nested classes. Here defines some
    transforms that only affect the image itself and others can be
    implemented in derived classes.

    Note that the format of the input sample is a dict where the values are
    often `numpy.ndarray' or `torch.Tensor'. The transforms is implemented
    with `cv2' and almost remain the same as `torchvision.transforms.transforms'
    except for `numpy.uint8' limitation of `Pillow'.
    """

    def __init__(self, pipeline):
        self.pipeline = self.Compose(pipeline)

    def __call__(self, sample):
        return self.pipeline(sample)

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, sample):
            for t in self.transforms:
                sample = t(sample)
            return sample

        def __repr__(self):
            format_string = self.__class__.__name__ + '('
            for t in self.transforms:
                format_string += '\n'
                format_string += '    {0}'.format(t)
            format_string += '\n)'
            return format_string

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, sample):
            sample['image'] = F.normalize(sample['image'], self.mean, self.std)
            return sample

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    class ColorJitter:
        """Randomly change the brightness, contrast and saturation of an image.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        """

        def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
            self.brightness = self._check_input(brightness, 'brightness')
            self.contrast = self._check_input(contrast, 'contrast')
            self.saturation = self._check_input(saturation, 'saturation')
            self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                         clip_first_on_zero=False)

        @staticmethod
        def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
            if isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError("If {} is a single number, it must be non negative.".format(name))
                value = [center - value, center + value]
                if clip_first_on_zero:
                    value[0] = max(value[0], 0)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                if not bound[0] <= value[0] <= value[1] <= bound[1]:
                    raise ValueError("{} values should be between {}".format(name, bound))
            else:
                raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

            # if value is 0 or (1., 1.) for brightness/contrast/saturation
            # or (0., 0.) for hue, do nothing
            if value[0] == value[1] == center:
                value = None
            return value

        @staticmethod
        def get_params(brightness, contrast, saturation, hue):
            transforms = []

            if brightness is not None:
                brightness_factor = random.uniform(brightness[0], brightness[1])
                transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

            if contrast is not None:
                contrast_factor = random.uniform(contrast[0], contrast[1])
                transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

            if saturation is not None:
                saturation_factor = random.uniform(saturation[0], saturation[1])
                transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

            if hue is not None:
                hue_factor = random.uniform(hue[0], hue[1])
                transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

            random.shuffle(transforms)
            transform = Compose(transforms)

            return transform

        def __call__(self, sample):
            transform = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            sample['image'] = transform(sample['image'])
            return sample

        def __repr__(self):
            format_string = self.__class__.__name__ + '('
            format_string += 'brightness={0}'.format(self.brightness)
            format_string += ', contrast={0}'.format(self.contrast)
            format_string += ', saturation={0}'.format(self.saturation)
            format_string += ', hue={0})'.format(self.hue)
            return format_string


class COCOTransform(BaseTransform):
    def __init__(self, pipeline):
        super(COCOTransform, self).__init__(pipeline)

    class ToTensor:
        def __call__(self, sample):
            sample['image'] = torch.from_numpy(
                np.ascontiguousarray(sample['image'].transpose((2, 0, 1)))).float()
            shuffle = torch.randperm(sample['bbox'].shape[0])
            sample['bbox'] = torch.from_numpy(sample['bbox']).float()[shuffle]
            sample['cls'] = torch.from_numpy(sample['cls']).long()[shuffle]
            if 'mask' in sample:
                sample['mask'] = torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(mask)) > 0 for mask in sample['mask']])[shuffle] \
                    if len(sample['mask']) > 0 else torch.empty(0, *sample['image'].shape[1:]) > 0
            return sample

        def __repr__(self):
            return self.__class__.__name__ + '()'

    class RandomCrop:
        def __init__(self, p=0.5, image_min_iou=0.64, bbox_min_iou=0.64):
            self.p = p
            self.image_min_iou = image_min_iou
            self.bbox_min_iou = bbox_min_iou
            self.image_max_ratio = image_min_iou ** 0.5
            self.bbox_max_ratio = bbox_min_iou ** 0.5

        def __call__(self, sample):
            if random.random() < self.p:
                height, width = sample['image'].shape[:2]

                if sample['bbox'].shape[0] == 0:
                    left = int(random.uniform(0, width * (1 - self.image_max_ratio)) + 0.5)
                    right = int(random.uniform(width * self.image_max_ratio, width) + 0.5)
                    top = int(random.uniform(0, height * (1 - self.image_max_ratio)) + 0.5)
                    down = int(random.uniform(height * self.image_max_ratio, height) + 0.5)
                else:
                    bx, by, bw, bh = np.split(sample['bbox'], 4, axis=1)
                    bx1 = (bx - bw / 2) * width
                    bx2 = (bx + bw / 2) * width
                    by1 = (by - bh / 2) * height
                    by2 = (by + bh / 2) * height

                    bbox_left = (bx1 * self.bbox_max_ratio + bx2 * (1 - self.bbox_max_ratio)).min()
                    bbox_right = (bx1 * (1 - self.bbox_max_ratio) + bx2 * self.bbox_max_ratio).max()
                    bbox_top = (by1 * self.bbox_max_ratio + by2 * (1 - self.bbox_max_ratio)).min()
                    bbox_down = (by1 * (1 - self.bbox_max_ratio) + by2 * self.bbox_max_ratio).max()

                    left = int(random.uniform(0, min(bbox_left, width * (1 - self.image_max_ratio))) + 0.5)
                    right = int(random.uniform(max(bbox_right, width * self.image_max_ratio), width) + 0.5)
                    top = int(random.uniform(0, min(bbox_top, height * (1 - self.image_max_ratio))) + 0.5)
                    down = int(random.uniform(max(bbox_down, height * self.image_max_ratio), height) + 0.5)

                    bx1_new = np.maximum(bx1 - left, 0)
                    bx2_new = np.minimum(bx2 - left, right - left + 1)
                    by1_new = np.maximum(by1 - top, 0)
                    by2_new = np.minimum(by2 - top, down - top + 1)

                    width_new = right - left + 1
                    height_new = down - top + 1
                    bx_new = (bx1_new + bx2_new) / 2 / width_new
                    by_new = (by1_new + by2_new) / 2 / height_new
                    bw_new = (bx2_new - bx1_new) / width_new
                    bh_new = (by2_new - by1_new) / height_new

                    sample['bbox'] = np.hstack([bx_new, by_new, bw_new, bh_new])

                sample['image'] = sample['image'][top:down+1, left:right+1]
                if 'mask' in sample:
                    sample['mask'] = [mask[top:down+1, left:right+1] for mask in sample['mask']]
                if 'info' in sample:
                    sample['info']['crop'] = (top, down+1, left, right+1) + (height, width)

            return sample

        def __repr__(self):
            return self.__class__.__name__ + '(p={0}, image_min_iou={1}, bbox_min_iou={2})'\
                .format(self.p, self.image_min_iou, self.bbox_min_iou)

    class Resize:
        """Random resize with padding

        Args:
            size (int, list, tuple): resized shape of output
            interpolation (str): methods provided by cv2 ('nearest', 'linear',
                'area', 'cubic', 'lanczos4')
            pad_needed (bool): if false, directly warp the image and masks,
                otherwise, perform random warp and padding before resize
            warp_p (float): the probability to directly warp the image
            jitter (float): aspect ratio jitter
            random_place (bool): if false, place the image at the center place,
                otherwise, random select the place in the output image space
            pad_p (float): the probability to add extra padding for the longer
                side compared with the random aspect ratio
            pad_ratio (float): the maximum ratio of random longer side padding
            pad_value (float, list, tuple): padded value
        """
        def __init__(self, size, interpolation='linear', pad_needed=True, warp_p=0., jitter=0.,
                     random_place=False, pad_p=0., pad_ratio=0., pad_value=255/2):
            assert isinstance(size, int) or len(size) == 2
            self.size = tuple(_pair(size))
            self.aspect_ratio = self.size[1] / self.size[0]
            self.interpolation = _str_to_cv2_interpolation[interpolation]
            self.pad_needed = pad_needed
            self.warp_p = warp_p
            self.jitter = jitter
            self.random_place = random_place
            self.pad_p = pad_p
            self.pad_ratio = pad_ratio
            self.pad_value = pad_value

        def __call__(self, sample):
            h, w = self.size
            if self.pad_needed and random.random() > self.warp_p:
                oh, ow = sample['image'].shape[:2]
                dh, dw = oh * self.jitter, ow * self.jitter
                new_aspect_ratio = (ow + random.uniform(-dw, dw)) / (oh + random.uniform(-dh, dh))
                if new_aspect_ratio < self.aspect_ratio:
                    nh = int(h * (1 - random.uniform(0, self.pad_ratio)) + 0.5) \
                        if random.random() < self.pad_p else h
                    nw = int(nh * new_aspect_ratio + 0.5)
                else:
                    nw = int(w * (1 - random.uniform(0, self.pad_ratio)) + 0.5) \
                        if random.random() < self.pad_p else w
                    nh = int(nw / new_aspect_ratio + 0.5)

                pad_left = int(random.uniform(0, w - nw) + 0.5) if self.random_place else int((w - nw) / 2 + 0.5)
                pad_top = int(random.uniform(0, h - nh) + 0.5) if self.random_place else int((h - nh) / 2 + 0.5)
                pad_right = w - nw - pad_left
                pad_down = h - nh - pad_top

                sample['bbox'][:, 0] = (sample['bbox'][:, 0] * nw + pad_left) / w
                sample['bbox'][:, 1] = (sample['bbox'][:, 1] * nh + pad_top) / h
                sample['bbox'][:, 2] = sample['bbox'][:, 2] * nw / w
                sample['bbox'][:, 3] = sample['bbox'][:, 3] * nh / h

                padding = (pad_top, pad_down, pad_left, pad_right)
                sample['image'] = imresize(sample['image'], (nw, nh), self.interpolation)
                sample['image'] = impad(sample['image'], padding, self.pad_value)
                if 'mask' in sample:
                    sample['mask'] = [imresize(mask, (nw, nh), cv2.INTER_NEAREST) for mask in sample['mask']]
                    sample['mask'] = [impad(mask, padding, 0) for mask in sample['mask']]
                if 'info' in sample:
                    sample['info']['pad'] = padding + (h, w)
            else:
                sample['image'] = imresize(sample['image'], (w, h), self.interpolation)
                if 'mask' in sample:
                    sample['mask'] = [imresize(mask, (w, h), cv2.INTER_NEAREST) for mask in sample['mask']]

            return sample

        def __repr__(self):
            interpolate_str = _cv2_interpolation_to_str[self.interpolation]
            return self.__class__.__name__ + '(size={0}, interpolation={1}, pad_needed={2}, ' \
                                             'warp_p={3}, jitter={4}, random_place={5}, ' \
                                             'pad_p={6}, pad_ratio={7}, pad_value={8})' \
                .format(self.size, interpolate_str, self.pad_needed, self.warp_p, self.jitter,
                        self.random_place, self.pad_p, self.pad_ratio, self.pad_value)

    class RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, sample):
            if random.random() < self.p:
                sample['image'] = imhflip(sample['image'])
                sample['bbox'][:, 0] = 1 - sample['bbox'][:, 0]
                if 'mask' in sample:
                    sample['mask'] = [imhflip(mask) for mask in sample['mask']]
                if 'info' in sample:
                    sample['info']['hflip'] = True
            return sample

        def __repr__(self):
            return self.__class__.__name__ + '(p={})'.format(self.p)

    class RandomVerticalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, sample):
            if random.random() < self.p:
                sample['image'] = imvflip(sample['image'])
                sample['bbox'][:, 1] = 1 - sample['bbox'][:, 1]
                if 'mask' in sample:
                    sample['mask'] = [imvflip(mask) for mask in sample['mask']]
                if 'info' in sample:
                    sample['info']['vflip'] = True
            return sample

        def __repr__(self):
            return self.__class__.__name__ + '(p={})'.format(self.p)

    class ShortEdgeResize:
        def __init__(self, short_length, max_size, interpolation='linear'):
            self.short_length = short_length
            self.max_size = max_size
            self.interpolation = _str_to_cv2_interpolation[interpolation]

        def __call__(self, sample):
            h, w = sample['image'].shape[:2]
            size = np.random.choice(self.short_length)
            scale = min(size / min(h, w), self.max_size / max(h, w))
            nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
            sample['image'] = imresize(sample['image'], (nw, nh), self.interpolation)
            if 'mask' in sample:
                sample['mask'] = [imresize(mask, (nw, nh), cv2.INTER_NEAREST) for mask in sample['mask']]
            return sample

        def __repr__(self):
            return self.__class__.__name__ + \
                   '(short_length={}, max_size={}, interpolation={})'.format(
                       self.short_length, self.max_size, self.interpolation)

    class Pad:
        def __init__(self, size_divisor=32, pad_value=255/2):
            self.size_divisor = size_divisor
            self.pad_value = pad_value

        def __call__(self, sample):
            height, width = sample['image'].shape[:2]
            new_height = int(math.ceil(height / self.size_divisor) * self.size_divisor)
            new_width = int(math.ceil(width / self.size_divisor) * self.size_divisor)
            pad_left, pad_top = (new_width - width) // 2, (new_height - height) // 2
            pad_right, pad_down = new_width - width - pad_left, new_height - height - pad_top

            sample['bbox'][:, 0] = (sample['bbox'][:, 0] * width + pad_left) / new_width
            sample['bbox'][:, 1] = (sample['bbox'][:, 1] * height + pad_top) / new_height
            sample['bbox'][:, 2] = sample['bbox'][:, 2] * width / new_width
            sample['bbox'][:, 3] = sample['bbox'][:, 3] * height / new_height

            padding = (pad_top, pad_down, pad_left, pad_right)
            sample['image'] = impad(sample['image'], padding, self.pad_value)
            if 'mask' in sample:
                sample['mask'] = [impad(mask, padding, 0) for mask in sample['mask']]
            if 'info' in sample:
                sample['info']['pad'] = padding + (new_height, new_width)

            return sample

        def __repr__(self):
            return self.__class__.__name__ + '(size_divisor={}, pad_value={})'.format(
                self.size_divisor, self.pad_value)


class FastCOCOTransform(BaseTransform):
    """A fast and simplified version that can execute transforms in GPU

    It's recommended for fast inference or speed benchmark. The input should be a
    float `torch.Tensor' with 4 dimensions (n, h, w, c).
    """
    def __init__(self, pipeline, use_cuda=True):
        super(FastCOCOTransform, self).__init__(pipeline)
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def __call__(self, image):
        # image float tensor with shape (n, h, w, c)
        if self.device != image.device:
            image = image.to(self.device)
        image = image.permute(0, 3, 1, 2).contiguous()
        image = self.pipeline(image)
        return image

    class Resize:
        def __init__(self, size, interpolation='bilinear', align_corners=False):
            assert isinstance(size, int) or len(size) == 2
            self.size = tuple(_pair(size))
            self.interpolation = interpolation
            self.align_corners = align_corners

        def __call__(self, image):
            image = nn.functional.interpolate(image, size=self.size, mode=self.interpolation,
                                              align_corners=self.align_corners)
            return image

        def __repr__(self):
            return self.__class__.__name__ + '(size={0}, interpolation={1}, align_corners={2})' \
                .format(self.size, self.interpolation, self.align_corners)

    class ShortEdgeResize:
        def __init__(self, short_length, max_size, interpolation='bilinear', align_corners=False):
            self.short_length = short_length
            self.max_size = max_size
            self.interpolation = interpolation
            self.align_corners = align_corners

        def __call__(self, image):
            h, w = image.shape[-2:]
            scale = min(self.short_length / min(h, w), self.max_size / max(h, w))
            nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
            image = nn.functional.interpolate(image, size=(nh, nw), mode=self.interpolation,
                                              align_corners=self.align_corners)
            return image

        def __repr__(self):
            return self.__class__.__name__ + \
                   '(short_length={}, max_size={}, interpolation={}, align_corners={})'.format(
                       self.short_length, self.max_size, self.interpolation, self.align_corners)

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, image):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=image.device)
            std = torch.tensor(self.std, dtype=torch.float32, device=image.device)
            image.sub_(mean[:, None, None]).div_(std[:, None, None])
            return image

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
