import os
import cv2
import csv
import numpy as np
import configparser
import os.path as osp
from PIL import Image

from src.frcnn.config import config

from mindspore import Tensor


class MOTObjDetectDatasetGenerator:
    def __init__(self, root, height=768, width=1280, split_seqs=None, vis_threshold=0.25):
        self.root = root
        self._height = height
        self._width = width
        self._vis_threshold = vis_threshold
        self._img_paths = []
        self._split_seqs = split_seqs

        # set self._img_paths
        for f in sorted(os.listdir(root)):
            path = osp.join(root, f)
            if not osp.isdir(path):
                continue

            if split_seqs is not None and f not in split_seqs:
                continue

            config_file = osp.join(path, 'seqinfo.ini')

            assert osp.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config['Sequence']['seqLength'])
            im_ext = config['Sequence']['imExt']
            im_dir = config['Sequence']['imDir']

            img_dir = osp.join(path, im_dir)

            for i in range(seq_len):
                img_path = osp.join(img_dir, f"{i + 1:06d}{im_ext}")
                assert osp.exists(img_path), \
                    'Path does not exist: {img_path}'
                self._img_paths.append(img_path)

    def _get_annotation(self, idx):
        if 'test' in self.root:
            num_objs = 0
            boxes = Tensor(np.array([]))

            # boxes, labels
            return boxes, self.ones((num_objs,))

        img_path = self._img_paths[idx]
        file_index = int(osp.basename(img_path).split('.')[0])

        gt_file = osp.join(osp.dirname(osp.dirname(img_path)), 'gt', 'gt.txt')

        assert osp.exists(gt_file), 'GT file does not exists: {}'.format(gt_file)

        bounding_boxes = []

        with open(gt_file, 'r') as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                visibility = float(row[8])
                if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and visibility >= self._vis_threshold:
                    bb = {}
                    bb['bb_left'] = int(row[2])
                    bb['bb_top'] = int(row[3])
                    bb['bb_width'] = int(row[4])
                    bb['bb_height'] = int(row[5])
                    bb['visibility'] = float(row[8])

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1, y1 = bb['bb_left'] - 1, bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2, y2 = x1 + bb['bb_width'] - 1, y1 + bb['bb_height'] - 1

            boxes[i, 0], boxes[i, 1] = x1, y1
            boxes[i, 2], boxes[i, 3] = x2, y2
        return boxes, np.ones((num_objs,)), np.ones((num_objs,), dtype=np.bool_)

    def __getitem__(self, idx):
        img_path = self._img_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        w_scale = self._width / img.shape[1]
        h_scale = self._height / img.shape[0]
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

        boxes, labels, valid_num = self._get_annotation(idx)
        boxes = boxes * scale_factor
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self._width - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self._height - 1)

        img_shape = np.asarray((self._height, self._width, 1.0), dtype=np.float32)

        return img, img_shape, boxes, labels, valid_num

    def __len__(self):
        return len(self._img_paths)


def preprocess_fn(img, boxes, labels, valid_num, flip_ratio):
    max_number = 128

    boxes = np.pad(boxes, ((0, max_number - boxes.shape[0]), (0, 0)), mode="constant", constant_values=0)
    labels = np.pad(labels, ((0, max_number - labels.shape[0])), mode="constant", constant_values=-1)
    valid_num = np.pad(valid_num, ((0, max_number - valid_num.shape[0])), mode="constant", constant_values=False)

    """flip operation for image"""
    if np.random.rand() > flip_ratio:
        return img, boxes, labels, valid_num
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = boxes.copy()
    _, _, w = img_data.shape

    flipped[..., 0::4] = w - boxes[..., 2::4] - 1
    flipped[..., 2::4] = w - boxes[..., 0::4] - 1

    return img_data, flipped, labels, valid_num

