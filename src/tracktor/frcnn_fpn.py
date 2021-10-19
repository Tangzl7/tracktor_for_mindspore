from model_utils.config import config
from frcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet


class FRCNN_FPN(Faster_Rcnn_Resnet):

	def __init__(self, num_classes):
		super(FRCNN_FPN, self).__init__(config)
		self.original_image_sizes = None
		self.preprocessed_images = None
		self.features = None
