import cv2
import numpy as np
from collections import deque

from src.tracktor.util import clip_boxes

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

        self.reset()

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
            self.tracks.append(Track(ops.Reshape()(new_det_box[i], (1, -1)),
                                     self.track_num + i, self.inactive_patience, self.max_features_num))

        self.track_num += num_new

    """regress the position of the tracks and also checks their scores"""
    """todo"""

    def regress_tracks(self, blob):
        boxes_in = self.get_box()  # get the track's pos
        boxes, scores = self.obj_detect.predict_boxes((boxes_in, ))
        pos = clip_boxes(Tensor(boxes), Tensor(blob['im_info'][0][:2]))

        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            tmp_box = t.box.asnumpy().copy()
            scores[i] = 0.9
            tmp_box[0][-1] = scores[i]
            t.box = Tensor(tmp_box)
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                tmp_box_ = ops.Concat(-1)((pos[i], Tensor([scores[i]])))
                t.box = ops.Reshape()(tmp_box_, (-1, 5))

    """get the positions of all active tracks"""

    def get_box(self):
        if len(self.tracks) == 1:
            features = self.tracks[0].box
        elif len(self.tracks) > 1:
            features = ops.Concat(0)([t.box for t in self.tracks])
        else:
            features = Tensor(np.array([]))
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
            t.last_box = t.box

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['data'], blob['im_info'])
        boxes, cls = self.obj_detect.detect(blob['data'], blob['im_info'])
        boxes, cls = np.squeeze(boxes), np.squeeze(cls)
        # filter out tracks that have too low person score
        boxes = boxes[cls == 1, :]
        # for test
        # boxes = np.array(
        #     [[0.023, 0.032, 0.445, 0.664, 0.6], [0.023, 0.032, 0.445, 0.664, 0.5], [0.065, 0.021, 0.556, 0.785, 0.8],
        #      [0.032, 0.324, 0.232, 0.435, 0.2]])
        inds = ops.Greater()(Tensor(boxes[:, -1]), self.detection_person_thresh).asnumpy()

        det_pos = boxes[inds, :-1]
        det_score = boxes[inds, -1]

        ##################
        # Predict tracks #
        ##################
        num_tracks = 0
        nms_inp_reg = Tensor(np.array([]))
        if len(self.tracks):
            # regress
            self.regress_tracks(blob)

            if len(self.tracks):
                nms_inp_reg = self.get_box()
                nms_keep_det, _, nms_keep_masks = ops.NMSWithMask(self.regression_nms_thresh)(nms_inp_reg)
                nms_keep_det, nms_keep_masks = nms_keep_det.asnumpy(), nms_keep_masks.asnumpy()
                nms_keep_det = nms_keep_det[nms_keep_masks, :]
                for i in range(len(nms_inp_reg)):
                    if nms_inp_reg[i].asnumpy() not in nms_keep_det:
                        self.tracks_to_inactive(self.tracks[i])
                nms_inp_reg = Tensor(nms_keep_det)

        #####################
        # Create new tracks #
        #####################
        nms_inp_det = boxes[inds, :]
        if len(nms_inp_det) > 0:
            nms_inp_det, _, masks = ops.NMSWithMask(self.detection_nms_thresh)(
                Tensor(nms_inp_det, dtype=ms.dtype.float32))
            nms_inp_det, masks = nms_inp_det.asnumpy(), masks.asnumpy()
            nms_inp_det = Tensor(nms_inp_det[masks, :])
            # check with every track in a single run (problem if tracks delete each other)
            for i in range(num_tracks):
                nms_inp = ops.Concat()(nms_inp_reg[i], nms_inp_det)
                nms_keep_det, _, nms_keep_masks = ops.NMSWithMask(self.detection_nms_thresh)(nms_inp)
                for j in range(len(nms_keep_masks)):
                    if nms_keep_masks[j] == True and nms_keep_det[j] == nms_inp_reg[i]:
                        nms_keep_masks[j] = False
                        break
                nms_keep_det, nms_keep_masks = nms_keep_det.asnumpy(), nms_keep_masks.asnumpy()
                nms_inp_det = Tensor(nms_keep_det[nms_keep_masks, :])

        if len(nms_inp_det) > 0:
            # add new track
            if len(nms_inp_det) > 0:
                self.add(nms_inp_det)

        ####################
        # Generate Results #
        ####################
        for t in self.tracks:
            track_ind = int(t.id)
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            box = t.box[0]
            self.results[track_ind][self.im_index] = box.asnumpy()

        self.im_index += 1
        self.last_image = blob['data'][0]
        self.clear_inactive()


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
