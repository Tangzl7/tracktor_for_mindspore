import numpy as np

import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
import mindspore.ops as ops

# from src.frcnn.config import config
from src.frcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50


class FRCNN_FPN(Faster_Rcnn_Resnet50):

	def __init__(self, config):
		super(FRCNN_FPN, self).__init__(config)
		self.original_image_sizes = None
		self.preprocessed_images = None
		self.features = None
		self.ones = ops.Ones()

	def detect(self, img_data, img_metas):
		return self(img_data, img_metas)

	def predict_boxes(self, boxes):
		# proposal: tuple: batch, 1000*5
		# proposal_mask: tuple: batch, 1000*5
		proposal, proposal_mask = boxes, ()
		for i in proposal:
			proposal_mask = proposal_mask + (self.ones((1000, ), mstype.bool_))
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
		rois = F.stop_gradient(rois)
		# roi_feats: nums*256*7*7

		roi_feats = self.roi_align_test(rois,
										self.cast(self.features[0], mstype.float32),
										self.cast(self.features[1], mstype.float32),
										self.cast(self.features[2], mstype.float32),
										self.cast(self.features[3], mstype.float32))

		roi_feats = self.cast(roi_feats, self.ms_type)
		rcnn_masks = self.concat(mask_tuple)
		rcnn_masks = F.stop_gradient(rcnn_masks)
		rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
		# rcnn_cls_loss: cls layer result, (nums, 81)  rcnn_reg_loss: reg layer result, (nums, 81*4)
		rcnn_loss, rcnn_cls_loss, rcnn_reg_loss, _ = self.rcnn(roi_feats,
															   bbox_targets,
															   rcnn_labels,
															   rcnn_mask_squeeze)

		output = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all, self.img_metas)

		return output

	def load_image(self, img_data, img_metas):
		self.set_train(False)
		self.img_metas = img_metas
		self.preprocessed_image = img_data
		self.features = self.backbone(img_data)
		self.features = self.fpn_ncek(self.features)

# if __name__ == '__main__':
# 	frcnn = Faster_Rcnn_Resnet50(config)
# 	img = Tensor(np.random.random((1, 3, 768, 1280)), dtype=ms.dtype.float32)
# 	img_meta = Tensor(np.array([[480, 640, 1.6, 1.6]]), dtype=ms.dtype.float32)
# 	frcnn.set_train(False)
# 	output = frcnn(img, img_meta, 0, 0, 0)
# 	print(output)
