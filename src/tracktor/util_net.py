import mindspore as ms
from mindspore import nn
import mindspore.ops as ops


class NMSWithMask(nn.Cell):
    def __init__(self, thresh):
        super(NMSWithMask, self).__init__()
        self.nms = ops.NMSWithMask(thresh)

    def construct(self, x):
        return self.nms(x)


class InitProposal(nn.Cell):
    def __init__(self):
        super(InitProposal, self).__init__()
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.concat = ops.Concat()
        self.rpn_max_num = 1000

    def construct(self, x):
        proposal, proposal_mask = (), ()
        for i in range(len(x)):
            boxes_false = self.zeros((self.rpn_max_num - len(x[i]), 5), ms.float32)
            proposal = proposal + (self.concat((x[i], boxes_false)),)
            masks_true = self.ones((len(x[i])), ms.bool_)
            masks_false = self.zeros((self.rpn_max_num - len(x[i])), ms.bool_)
            proposal_mask = proposal_mask + (self.concat((masks_true, masks_false)),)
        return proposal, proposal_mask