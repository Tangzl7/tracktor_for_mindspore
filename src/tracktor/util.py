import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


def clip_boxes(boxes, size):
    """
    Clip boxes to image boundaries.
    """
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = ops.clip_by_value(boxes_x, Tensor(0), width)
    boxes_y = ops.clip_by_value(boxes_y, Tensor(0), height)

    clipped_boxes = ops.Stack(-1)((boxes_x, boxes_y))
    return ops.Reshape()(clipped_boxes, (-1, 4))