import numpy as np

import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

from src.frcnn.model_utils.config import config
from src.frcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50


class FRCNN_FPN(Faster_Rcnn_Resnet50):

	def __init__(self, config):
		super(FRCNN_FPN, self).__init__(config)

	def detect(self, img_data, img_metas):
		if isinstance(img_data, np.ndarray):
			img_data = Tensor(img_data)
		if isinstance(img_metas, np.ndarray):
			img_metas = Tensor(img_metas)

		output = self(img_data, img_metas)

		pred_boxes, pred_cls, pred_mask = output[0].asnumpy(), output[1].asnumpy(), output[2].asnumpy()
		result_boxes, result_cls = [], []

		for j in range(self.config.test_batch_size):
			pred_boxes_j = np.squeeze(pred_boxes[j, :, :])
			pred_cls_j = np.squeeze(pred_cls[j, :, :])
			pred_mask_j = np.squeeze(pred_mask[j, :, :])
			result_boxes.append(pred_boxes_j[pred_mask_j, :])
			result_cls.append(pred_cls_j[pred_mask_j])

		return np.array(result_boxes), np.array(result_cls)

	def construct(self, proposal, proposal_mask, img_metas, features):
		bboxes_tuple = ()
		mask_tuple = ()
		# mask_tuple: tuple: batch, (1000, )
		mask_tuple += proposal_mask
		bbox_targets = proposal_mask
		rcnn_labels = proposal_mask
		# bboxes_tuple: tuple: batch, (1000, 4)
		for p_i in proposal:
			bboxes_tuple += (p_i[::, 0:4:1],)

		if self.test_batch_size > 1:
			bboxes_all = self.concat(bboxes_tuple)
		else:
			bboxes_all = bboxes_tuple[0]
		if self.device_type == "Ascend":
			bboxes_all = self.cast(bboxes_all, mstype.float16)
		rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))
		# rois: (batch*1000, 5)
		rois = self.cast(rois, mstype.float32)
		# roi_feats: nums*256*7*7

		roi_feats = self.roi_align_test(rois,
										self.cast(features[0], mstype.float32),
										self.cast(features[1], mstype.float32),
										self.cast(features[2], mstype.float32),
										self.cast(features[3], mstype.float32))

		roi_feats = self.cast(roi_feats, self.ms_type)
		rcnn_masks = self.concat(mask_tuple)
		rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
		# rcnn_cls_loss: cls layer result, (nums, 81)  rcnn_reg_loss: reg layer result, (nums, 81*4)
		rcnn_loss, rcnn_cls_loss, rcnn_reg_loss, _ = self.rcnn(roi_feats,
															   bbox_targets,
															   rcnn_labels,
															   rcnn_mask_squeeze)

		output = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all, img_metas, True)

		return output[0][1], output[1], output[2]


class FRCNN_FPN_Fea(Faster_Rcnn_Resnet50):
	def __init__(self, config):
		super(FRCNN_FPN_Fea, self).__init__(config)

	def construct(self, img_data):
		features = self.backbone(img_data)
		features = self.fpn_ncek(features)
		return features


if __name__ == '__main__':
	frcnn = FRCNN_FPN(config)
	img = Tensor(np.random.random((1, 3, 768, 1280)), dtype=ms.dtype.float32)
	img_meta = Tensor(np.array([[480, 640, 1.6, 1.6]]), dtype=ms.dtype.float32)
	frcnn.set_train(False)
	output = frcnn(img, img_meta, 0, 0, 0)
	print(output)
	# frcnn = FRCNN_FPN(config)
	# img = Tensor(np.random.random((1, 3, 768, 1280)), dtype=ms.dtype.float32)
	# img_meta = Tensor(np.array([[480, 640, 1.6, 1.6]]), dtype=ms.dtype.float32)
	# frcnn.load_image(img, img_meta)
	# box = Tensor(np.random.random((50, 5)), dtype=ms.dtype.float32)
	# box = (box, )
	# frcnn.predict_boxes(box)

