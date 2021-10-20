from src.config import config
from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50


class FRCNN_FPN(Faster_Rcnn_Resnet50):

	def __init__(self, num_classes):
		super(FRCNN_FPN, self).__init__(config)
		self.original_image_sizes = None
		self.preprocessed_images = None
		self.features = None
