import cv2
import numpy as np
from collections import deque

from src.tracktor.utils import clip_boxes, resize_boxes

import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
from sklearn.metrics import pairwise_distances


class Tracker():
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, tracker_cfg):
        self.obj_detect = obj_detect  # object detector
        # self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']  # 10
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model = tracker_cfg['motion_model']

        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.det_nms = ops.NMSWithMask(self.detection_nms_thresh)
        self.reg_nms = ops.NMSWithMask(self.regression_nms_thresh)

    def reset(self, hard=True):
        self.tracks = []  # active target trajectory set
        self.inactive_tracks = []  # inactive target trajectory set

        if hard:
            self.track_num = 0  # track_id
            self.results = {}
            self.im_index = 0

    """remove some tracks"""

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.box = t.last_box
        self.inactive_tracks += tracks

    """initializes new track objects and saves them"""

    def add(self, new_det_box):
        num_new = len(new_det_box)
        for i in range(num_new):
            self.tracks.append(Track(new_det_box[i].reshape((1, -1)),
                                     self.track_num + i, self.inactive_patience, self.max_features_num))

        self.track_num += num_new

    """regress the position of the tracks and also checks their scores"""
    """todo"""

    def regress_tracks(self, blob):
        boxes_in = self.get_box()  # get the track's pos
        boxes_in = resize_boxes(boxes_in, blob['img_shape'][0][:2], blob['img'].shape[-2:])
        poses, scores = self.obj_detect.predict_boxes((Tensor(boxes_in),))
        boxes = np.concatenate((poses, np.expand_dims(scores, axis=-1)), axis=1)
        boxes = clip_boxes(boxes, blob['img_shape'][0][:2])

        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.box[0][-1] = boxes[i, -1]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                t.box[0] = boxes[i]

    """get the positions of all active tracks"""

    def get_box(self):
        if len(self.tracks) == 1:
            features = self.tracks[0].box
        elif len(self.tracks) > 1:
            features = np.concatenate([t.box for t in self.tracks])
        else:
            features = np.array([])
        return features

    """check if inactive tracks should be removed"""

    def clear_inactive(self):
        to_remove = []
        for t in self.inactive_tracks:
            if t.is_to_purge():
                to_remove.append(t)
        for t in to_remove:
            self.inactive_tracks.remove(t)

    """this function should be called every timestep to perform tracking with a blob 
    containing the image information
    """

    def step(self, blob):
        for t in self.tracks:
            t.last_box = t.box.copy()

        # img = Tensor(np.ones([1, 3, 768, 1280]), dtype=ms.float32)
        # blob['img'] = img

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['img'], blob['img_shape'])
        if self.public_detections:
            dets = Tensor(blob['dets'])
            if len(dets) > 0:
                poses, scores = self.obj_detect.predict_boxes((dets,))
                boxes = np.concatenate((poses, np.expand_dims(scores, axis=-1)), axis=1)
            else:
                boxes = np.array([])
        else:
            boxes, cls = self.obj_detect.detect(blob['img'], blob['img_shape'])
            boxes, cls = np.squeeze(boxes), np.squeeze(cls)
            boxes = boxes[cls == 0, :]

        if len(boxes) > 0:
            boxes = clip_boxes(boxes, blob['img_shape'][0][:2])
            inds = np.greater(boxes[:, -1], self.detection_person_thresh)
        else:
            inds = np.array([])

        if len(inds) > 0:
            det_boxes = boxes[inds]
        else:
            det_boxes = np.array([])

        ##################
        # Predict tracks #
        ##################
        if len(self.tracks):
            # regress
            self.regress_tracks(blob)

            if len(self.tracks):
                nms_inp_reg = self.get_box()
                nms_keep_det, _, nms_keep_masks = self.reg_nms(Tensor(nms_inp_reg, dtype=ms.dtype.float32))
                nms_keep_det, nms_keep_masks = nms_keep_det.asnumpy(), nms_keep_masks.asnumpy()
                nms_keep_det = nms_keep_det[nms_keep_masks, :]
                tracks_remove = []
                for i in range(len(nms_inp_reg)):
                    if nms_inp_reg[i] not in nms_keep_det:
                        tracks_remove.append(self.tracks[i])
                self.tracks_to_inactive(tracks_remove)

        #####################
        # Create new tracks #
        #####################
        if len(det_boxes) > 0:
            det_boxes, _, masks = self.det_nms(Tensor(det_boxes, dtype=ms.dtype.float32))
            det_boxes, masks = det_boxes.asnumpy(), masks.asnumpy()
            det_boxes = det_boxes[masks, :]
            # check with every track in a single run (problem if tracks delete each other)
            for i in range(len(self.tracks)):
                track_box = self.tracks[i].box.copy()
                track_box[0][-1] = 2.0
                nms_track_boxes = np.concatenate((track_box, det_boxes))
                nms_keep_det, _, nms_keep_masks = self.det_nms(Tensor(nms_track_boxes, dtype=ms.dtype.float32))
                nms_keep_det, nms_keep_masks = nms_keep_det.asnumpy(), nms_keep_masks.asnumpy()
                nms_keep_masks[0] = False
                det_boxes = nms_keep_det[nms_keep_masks, :]
                if len(det_boxes) == 0:
                    break

        if len(det_boxes) > 0:
            # add new track
            self.add(det_boxes)

        ####################
        # Generate Results #
        ####################
        for t in self.tracks:
            track_ind = int(t.id)
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            box = t.box[0]
            self.results[track_ind][self.im_index] = box.copy()

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

    def get_result(self):
        return self.results


class Track(object):
    """track class for every individual track."""

    def __init__(self, box, track_id, inactive_patience, max_features_num):
        self.id = track_id
        self.box = box
        self.features = deque([])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_box = Tensor(np.array([]))
        self.last_v = Tensor(np.array([]))
        self.gt_id = None

    """tests if the object has been too long inactive and is to remove"""

    def has_positive_area(self):
        return self.box[0][2] > self.box[0][0] and self.box[0][3] > self.box[0][1]

    def is_to_purge(self):
        self.count_inactive += 1
        self.last_box = Tensor(np.array([]))
        if self.count_inactive > self.inactive_patience:
            return True
        else:
            return False

    """add new appearance features to the object"""

    def add_features(self, features):
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    """compares test_features to features of this track object"""

    def test_features(self, test_features):
        if len(self.features) > 1:
            features = ops.Concat()(list(self.features))
        else:
            features = self.features[0]
        features = features.mean(0, keep_dims=True)
        features, test_features = features.asnumpy(), test_features.asnumpy()
        dist = pairwise_distances(features, test_features)
        return Tensor(dist)


if __name__ == '__main__':
    a = Tensor(np.array([]))
    a = Tensor(np.array([[1, 2], [3, 4]]))
    print(a)
