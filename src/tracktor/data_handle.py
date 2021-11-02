import cv2
import numpy as np
from PIL import Image

from src.tracktor.config import cfg

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# transforms_list = [py_vision.ToTensor(), c_vision.Normalize(normalize_mean, normalize_std)]

def data_process(frame):
  im = np.array(frame)

  data_, im_scales = _get_image_blob(frame)
  data = []
  for i in range(len(data_)):
    data.append(c_vision.HWC2CHW()(data_[i]))
  c = np.array([im.shape[0], im.shape[1], im_scales[0], im_scales[0]], dtype=np.float32)
  c = Tensor(c, dtype=ms.dtype.float32)
  data = Tensor(np.array(data), dtype=ms.dtype.float32)

  sample = {}
  sample['data'] = data
  sample['im_info'] = ops.ExpandDims()(c, 0)
  im = im / 255.0
  im = c_vision.Normalize(normalize_mean, normalize_std)(im)
  im = c_vision.HWC2CHW()(im)
  im = ops.ExpandDims()(Tensor(im), 0)
  sample['app_data'] = im
  return sample

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, (1280, 768), fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob