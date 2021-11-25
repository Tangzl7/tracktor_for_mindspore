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

    def _get_annotation_eval(self, idx):
        if 'test' in self.root:
            num_objs = 0
            boxes = Tensor(np.array([]))

            return {'boxes': boxes,
                    'labels': np.ones((num_objs,), dtype=np.int32),
                    'image_id': np.array([idx]),
                    'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                    'iscrowd': np.zeros((num_objs,), dtype=np.int32),
                    'visibilities': np.zeros((num_objs), dtype=np.float32)}

        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split('.')[0])

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
        visibilities = np.zeros((num_objs), dtype=np.float32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left'] - 1
            y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width'] - 1
            y2 = y1 + bb['bb_height'] - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']

        return {'boxes': boxes,
                'labels': np.ones((num_objs,), dtype=np.int32),
                'image_id': np.array([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': np.zeros((num_objs,), dtype=np.int32),
                'visibilities': visibilities, }

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
        return boxes, np.ones((num_objs,), dtype=np.float32), np.ones((num_objs,), dtype=np.bool_), np.array([idx])

    def imnormalize_column(self, img):
        """imnormalize operation for image"""
        mean = np.asarray([123.675, 116.28, 103.53])
        std = np.asarray([58.395, 57.12, 57.375])
        img_data = img.copy().astype(np.float32)
        cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
        cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
        cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

        img_data = img_data.astype(np.float32)
        return img_data

    def transpose_column(self, img):
        """transpose operation for image"""
        img_data = img.transpose(2, 0, 1).copy()
        img_data = img_data.astype(np.float32)

        return img_data

    def expand_img(self, img):
        expanded_img = np.zeros([3, self._height, self._width], dtype=np.float32)
        expanded_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img.copy()
        return expanded_img

    def __getitem__(self, idx):
        img_path = self._img_paths[idx]
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        img = img_rgb.copy()
        img[:, :, 0] = img_rgb[:, :, 2]
        img[:, :, 1] = img_rgb[:, :, 1]
        img[:, :, 2] = img_rgb[:, :, 0]
        # w_scale = self._width / img.shape[1]
        # h_scale = self._height / img.shape[0]
        img_shape = np.array((img.shape[0], img.shape[1]))
        scale_factor_ = min(max(self._height, self._width) / max(img.shape[0], img.shape[1]),
                            min(self._height, self._width) / min(img.shape[0], img.shape[1]))
        # scale_factor_ = [self._height / img.shape[0], self._width / img.shape[1]]
        scale_factor = np.array(
            [scale_factor_, scale_factor_, scale_factor_, scale_factor_], dtype=np.float32)
        img = cv2.resize(img, (int(img.shape[1]*scale_factor_), int(img.shape[0]*scale_factor_)), interpolation=cv2.INTER_LINEAR)

        boxes, labels, valid_num, image_id = self._get_annotation(idx)
        boxes = boxes * scale_factor
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, int(img.shape[1]*scale_factor_)-1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, int(img.shape[0]*scale_factor_)-1)

        # img_shape = np.asarray((self._height, self._width, 1.0), dtype=np.float32)
        img_shape = np.append(img_shape, (scale_factor_, scale_factor_))
        img_shape = np.asarray(img_shape, dtype=np.float32)

        img = self.imnormalize_column(img)
        img = self.transpose_column(img)
        img = self.expand_img(img)

        return img, img_shape, boxes, labels, valid_num, image_id

    def __len__(self):
        return len(self._img_paths)

    def write_results_files(self, results, output_dir):
        """Write the detections in the format for MOT17Det sumbission

        all_boxes[image] = N x 5 array of detections in (x1, y1, x2, y2, score)

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT17-01.txt
        ./MOT17-02.txt
        ./MOT17-03.txt
        ./MOT17-04.txt
        ./MOT17-05.txt
        ./MOT17-06.txt
        ./MOT17-07.txt
        ./MOT17-08.txt
        ./MOT17-09.txt
        ./MOT17-10.txt
        ./MOT17-11.txt
        ./MOT17-12.txt
        ./MOT17-13.txt
        ./MOT17-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        files = {}
        for image_id, res in results.items():
            path = self._img_paths[int(image_id)]
            img1, name = osp.split(path)
            # get image number out of name
            frame = int(name.split('.')[0])
            # smth like /train/MOT17-09-FRCNN or /train/MOT17-09
            tmp = osp.dirname(img1)
            # get the folder name of the sequence and split it
            tmp = osp.basename(tmp).split('-')
            # Now get the output name of the file
            out = tmp[0]+'-'+tmp[1]+'.txt'
            outfile = osp.join(output_dir, out)

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, results, ovthresh=0.5):
        """Evaluates the detections (not official!!)

        all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
        """

        if 'test' in self.root:
            print('No GT data available for evaluation.')
            return

        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(len(self._img_paths))]
        fp = [[] for _ in range(len(self._img_paths))]

        npos = 0
        gt = []
        gt_found = []

        for idx in range(len(self._img_paths)):
            annotation = self._get_annotation_eval(idx)
            bbox = annotation['boxes'][annotation['visibilities'] > self._vis_threshold]
            found = np.zeros(bbox.shape[0])
            gt.append(bbox)
            gt_found.append(found)

            npos += found.shape[0]

        # Loop through all images
        # for res in results:
        for im_index, (im_gt, found) in enumerate(zip(gt, gt_found)):
            # Loop through dets an mark TPs and FPs

            # im_index = res['image_id'].item()
            # im_det = results['boxes']
            # annotation = self._get_annotation(im_index)
            # im_gt = annotation['boxes'][annotation['visibilities'].gt(0.5)].cpu().numpy()
            # found = np.zeros(im_gt.shape[0])

            im_det = results[str(im_index)]['boxes']

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf

                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                            (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                            (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.

            tp[im_index] = im_tp
            fp[im_index] = im_fp

        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]

        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)

        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s+i] = tp_im
                fp_flat[i:s+i] = fp_im
                i += s

        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        if len(tp):
            tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap
        else:
            tp, fp, prec, rec, ap = 0, 0, 0, 0, 0

        print(f"AP: {ap:.2f} Prec: {prec:.2f} Rec: {rec:.2f} TP: {tp} FP: {fp}")

        return {'AP': ap, 'precision': prec, 'recall': rec, 'TP': tp, 'FP': fp}


def preprocess_fn(img, img_shape, boxes, labels, valid_num, image_id, flip_ratio):
    max_number = 128

    boxes = np.pad(boxes, ((0, max_number - boxes.shape[0]), (0, 0)), mode="constant", constant_values=0)
    labels = np.pad(labels, ((0, max_number - labels.shape[0])), mode="constant", constant_values=-1)
    valid_num = np.pad(valid_num, ((0, max_number - valid_num.shape[0])), mode="constant", constant_values=False)

    """flip operation for image"""
    if np.random.rand() > flip_ratio:
        if flip_ratio == -1:
            return img, img_shape, boxes, labels, valid_num, image_id
        else:
            return img, np.asarray((config.img_height, config.img_width, 1.0), dtype=np.float32), boxes, labels, valid_num
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = boxes.copy()
    _, _, w = img_data.shape

    flipped[..., 0::4] = w - boxes[..., 2::4] - 1
    flipped[..., 2::4] = w - boxes[..., 0::4] - 1

    return img_data, np.asarray((config.img_height, config.img_width, 1.0), dtype=np.float32), flipped, labels, valid_num

