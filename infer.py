import os
import math
import json
import argparse

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as config_module
import model as model_module
import utils.visualizer as visualizer_module
import utils.timer as timer
from trainer.builder import build, build_transform, build_postprocess
from eval.coco_eval import COCOMetrics
from data.dataset import COCODataset


def pad(image, size_divisor=32, pad_value=0):
    height, width = image.shape[-2:]
    new_height = int(math.ceil(height / size_divisor) * size_divisor)
    new_width = int(math.ceil(width / size_divisor) * size_divisor)
    pad_left, pad_top = (new_width - width) // 2, (new_height - height) // 2
    pad_right, pad_down = new_width - width - pad_left, new_height - height - pad_top

    padding = [pad_left, pad_right, pad_top, pad_down]
    image = F.pad(image, padding, value=pad_value)
    pad_info = padding + [new_height, new_width]

    return image, pad_info


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='inference config (default: None)')
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='model weights to inference (default: None)')
    parser.add_argument('-i', '--image', default=None, type=str,
                        help='the path of an image to infer')
    parser.add_argument('-d', '--image_dir', default=None, type=str,
                        help='the base directory of images to infer')
    parser.add_argument('-l', '--image_list', default=None, type=str,
                        help='the list file containing all images to infer')
    parser.add_argument('-j', '--json_file', default=None, type=str,
                        help='the coco json file to infer and produce json output')
    parser.add_argument('-n', '--num_images', default=None, type=int,
                        help='the number of images to infer')
    parser.add_argument('-b', '--benchmark', default=None, action='store_true',
                        help='benchmark the inference speed')
    parser.add_argument('-v', '--visualize', default=False, action='store_true',
                        help='produce visualization result')
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='the path to save visualization result')
    parser.add_argument('-s', '--show', default=False, action='store_true',
                        help='show visualization result on the window')
    args = parser.parse_args()

    # Load config
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    else:
        config = getattr(config_module, args.config)

    # Device
    assert config['n_gpu'] <= 1, 'Multi-gpu test is not supported'
    use_cuda = config['n_gpu'] > 0
    if use_cuda:
        device = torch.device('cuda:0')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    else:
        device = torch.device('cpu')

    # Build model, transform and postprocess
    config['model']['pretrained'] = None
    model = build(config['model'], model_module).to(device)
    weights = torch.load(args.weights, map_location=device)
    weights = weights['state_dict'] if 'state_dict' in weights else weights
    model.load_state_dict(weights, strict=True)
    config['transform']['use_cuda'] = use_cuda
    transform = build_transform(config['transform'])
    postprocess = build_postprocess(config['postprocess'], device=device)
    visualizer = build(config['visualizer'], visualizer_module, device=device)

    # image files
    if args.image:
        file_names = [os.path.basename(args.image)]
        image_files = [args.image]
    elif args.json_file:
        json_images = json.load(open(args.json_file))['images']
        if args.num_images:
            json_images = json_images[:args.num_images]
        file_names = [json_image['file_name'] for json_image in json_images]
        image_files = [os.path.join(args.image_dir, file_name) for file_name in file_names]
        sample_infos = [{'height': json_image['height'],
                         'width': json_image['width'],
                         'id': json_image['id']} for json_image in json_images]
        coco_metrics = COCOMetrics(
            gt_file=None, cat2label=COCODataset.CAT2LABEL, with_mask=True,
            save_dir=args.output if args.output is not None else '.',
        )
    elif args.image_dir:
        if args.image_list:
            file_names = [file_name.strip() for file_name in open(args.image_list)]
        else:
            file_names = os.listdir(args.image_dir)
        if args.num_images:
            file_names = file_names[:args.num_images]
        image_files = [os.path.join(args.image_dir, file_name) for file_name in file_names]
    else:
        raise ValueError('Either image or image_dir should be given.')

    # output files
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_files = [os.path.join(args.output, file_name) for file_name in file_names]
    else:
        output_files = None

    # Timer init
    timer.reset()
    timer.cpu() if str(device) == 'cpu' else timer.cuda()

    with torch.no_grad():
        model.eval()
        # warmup
        if args.benchmark:
            src_image = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2RGB)
            src_image = torch.tensor(src_image, dtype=torch.float32).to(device)
            image = transform(src_image.unsqueeze(0))
            image, pad_info = pad(image)
            for _ in range(10):
                predictions = model(image)
                predictions = postprocess(predictions)[0]

        n_iter = len(image_files)
        src_images, images, pad_infos = [], [], []

        with timer.timer('Main Loop'):
            for idx, image_file in tqdm(enumerate(image_files), total=n_iter):
                # Load and transform image
                # print(os.path.basename(image_file))
                with timer.timer('Load data'):
                    src_image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
                    src_image = torch.tensor(src_image, dtype=torch.float32).to(device)
                    image = transform(src_image.unsqueeze(0))
                    image, pad_info = pad(image)

                # Network forward and postprocess
                with timer.timer('Forward & Postprocess'):
                    predictions = model(image)
                    predictions = postprocess(predictions)

                # Convert to coco format
                if args.json_file and args.output:
                    with timer.timer('Convert Format'):
                        sample_info = [dict(sample_infos[idx], collate_pad=pad_info)]
                        coco_format_dets = coco_metrics.to_coco_format(sample_info, predictions)
                        coco_metrics.update_results(coco_format_dets)

                # Visualizer
                if args.visualize:
                    with timer.timer('Visualize'):
                        show_image = visualizer(predictions[0], src_image, pad_info)
                        if args.show:
                            plt.imshow(show_image)
                        if args.output:
                            plt.imsave(output_files[idx], show_image)

    if args.json_file:
        with open(coco_metrics.bbox_pred_file, 'w') as handle:
            json.dump(coco_metrics.bbox_results, handle)
        with open(coco_metrics.segm_pred_file, 'w') as handle:
            json.dump(coco_metrics.segm_results, handle)

    # Speed Statistics
    # timer.log_elapsed_time()
    timer_log = timer.get_all_elapsed_time()
    duration = timer_log.pop('Main Loop')
    print('The inference takes {0} seconds.'.format(duration / 1000))
    print('The average inference time is %.2f ms (%.2f fps)' %
          (duration / len(image_files), 1000 * len(image_files) / duration))
    for key, value in timer_log.items():
        print('%s: %.2fms (%.2ffps)' % (key, value, 1000 / value))
