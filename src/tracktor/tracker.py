import numpy as np
from collections import deque

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
            t.pos = t.last_pos
        self.inactive_tracks += tracks

    """initializes new track objects and saves them"""
    def add(self, new_det_pos, new_det_scores):
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(ops.Reshape()(new_det_pos[i], (1, -1)), new_det_scores[i],
                                     self.track_num + i, self.inactive_patience, self.max_features_num))

        self.track_num += num_new

    """regress the position of the tracks and also checks their scores"""
    """todo"""
    def regress_tracks(self, blob):
        pos = self.get_pos()  # get the track's pos
        out = self.obj_detect(pos)
        boxes = out[0]
        pos = clip_boxes_to_image(boxes, blob['im_info'][0][:2])
        scores = boxes[:, -1]

        s = []
        for i in range(len(self.tracks)-1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                t.pos = ops.Reshape()(pos[i], (1, -1))
        return Tensor(np.array(s[::-1]))

    """get the positions of all active tracks"""
    def get_pos(self):
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = ops.Concat()([t.feature for t in self.tracks], 0)
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
            t.last_pos = t.pos

        ###########################
        # Look for new detections #
        ###########################
        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
        boxes, scores, _ = self.obj_detect.detect(blob['data'][0], blob['im_info'][0])
        boxes, scores = boxes.asnumpy(), scores.asnumpy()
        # filter out tracks that have too low person score
        inds = ops.Greater()(scores, self.detection_person_thresh).asnumpy()

        det_pos = boxes[inds]
        det_score = scores[inds]

        ##################
        # Predict tracks #
        ##################
        num_tracks = 0
        nms_inp_reg = Tensor(np.array([]))
        if len(self.tracks):
            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):







class Track(object):
    """track class for every individual track."""

    def __init__(self, pos, score, track_id, inactive_patience, max_features_num):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = Tensor(np.array([]))
        self.last_v = Tensor(np.array([]))
        self.gt_id = None

    """tests if the object has been too long inactive and is to remove"""
    def is_to_purge(self):
        self.count_inactive += 1
        self.last_pos = Tensor(np.array([]))
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
    a = Tensor(np.array([[1, 2], [3, 4]]))
    b = ops.Greater()(a, 2)
    print(len(a))