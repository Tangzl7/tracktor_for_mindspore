import os
import cv2
import csv
import numpy as np
import configparser
import os.path as osp
from PIL import Image

from mindspore import Tensor


class MOTSequence:
    def __init__(self, seq_name, mot_dir, vis_threshold=0.0, height=768, width=1280,):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self._height = height
        self._width = width

        self._mot_dir = mot_dir

        self._train_folders = os.listdir(osp.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(osp.join(self._mot_dir, 'test'))

        assert seq_name in self._train_folders + self._test_folders, \
            'Image set does not exist: {}'.format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img_rgb = np.array(Image.open(data['im_path']).convert("RGB"))
        img = img_rgb.copy()
        img[:, :, 0] = img_rgb[:, :, 2]
        img[:, :, 1] = img_rgb[:, :, 1]
        img[:, :, 2] = img_rgb[:, :, 0]
        # w_scale = self._width / img.shape[1]
        # h_scale = self._height / img.shape[0]
        img_shape = np.array((img.shape[0], img.shape[1]))
        scale_factor_ = min(max(self._height, self._width) / max(img.shape[0], img.shape[1]),
                            min(self._height, self._width) / min(img.shape[0], img.shape[1]))
        scale_factor = np.array(
            [scale_factor_, scale_factor_, scale_factor_, scale_factor_], dtype=np.float32)
        img = cv2.resize(img, (int(img.shape[1]*scale_factor_), int(img.shape[0]*scale_factor_)), interpolation=cv2.INTER_LINEAR)
        img_shape = np.append(img_shape, (scale_factor_, scale_factor_))
        img_shape = np.asarray(img_shape, dtype=np.float32)

        img = self.imnormalize_column(img)
        img = self.transpose_column(img)
        img = self.expand_img(img)

        dets = np.array(data['dets'].copy())
        dets[:, :-1] = dets[:, :-1] * scale_factor
        dets[:, 0::2] = np.clip(dets[:, 0::2], 0, int(img_rgb.shape[1]*scale_factor_)-1)
        dets[:, 1::2] = np.clip(dets[:, 1::2], 0, int(img_rgb.shape[0]*scale_factor_)-1)

        sample = {}
        sample['img'] = np.expand_dims(img, axis=0)
        sample['img_shape'] = np.expand_dims(img_shape, axis=0)
        # sample['dets'] = np.array([det for det in data['dets']])
        sample['dets'] = np.array(dets)
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def imnormalize_column(self, img):
        """imnormalize operation for image"""
        mean = np.asarray([123.675, 116.28, 103.53])
        std = np.asarray([58.395, 57.12, 57.375])
        img_data = img.copy().astype(np.float32)
        cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)
        cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)
        cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)

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

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)

        config_file = osp.join(seq_path, 'seqinfo.ini')
        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        visibility, boxes, dets = {}, {}, {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, 'r') as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        det_file = osp.join(seq_path, 'det', 'det.txt')
        if osp.exists(det_file):
            with open(det_file, 'r') as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        for i in range(1, seqLength + 1):
            im_path = osp.join(imDir, f"{i:06d}.jpg")
            sample = {'gt': boxes[i],
                      'im_path': im_path,
                      'vis': visibility[i],
                      'dets': dets[i]}
            total.append(sample)

        return total, no_gt

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(osp.join(output_dir, self._seq_name), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1 + 1,
                         y2 - y1 + 1,
                         -1, -1, -1, -1])

    def load_results(self, output_dir):
        file_path = osp.join(output_dir, self._seq_name)
        results = {}

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as of:
            csv_reader = csv.reader(of, delimiter=',')
            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if not track_id in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = [x1, y1, x2, y2]

        return results
