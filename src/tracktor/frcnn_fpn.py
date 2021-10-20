import numpy as np

import mindspore as ms
from mindspore import Model
from mindspore.common.tensor import Tensor

from src.config import config
from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.FasterRcnn.faster_rcnn_r50 import FasterRcnn_Infer


class FRCNN_FPN(Faster_Rcnn_Resnet50):

	def __init__(self, num_classes):
		super(FRCNN_FPN, self).__init__(config)
		self.original_image_sizes = None
		self.preprocessed_images = None
		self.features = None


if __name__ == '__main__':
	frcnn = FasterRcnn_Infer(config)
	img = Tensor(np.random.random((1, 3, 768, 1280)), dtype=ms.dtype.float32)
	img_meta = Tensor(np.array([[768, 1280, 1, 1]]), dtype=ms.dtype.float32)
	# net = Model(frcnn)
	frcnn = frcnn.set_train(mode=False)
	output = frcnn(img, img_meta)
	print(output)
