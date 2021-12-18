import os

from mindspore.train.callback import Callback


class EvalCallBack(Callback):
    def __init__(self, config, apply_eval, datasetsize, checkpoint_path):
        super(EvalCallBack, self).__init__()
        self.config = config
        self.apply_eval = apply_eval
        self.datasetsize = datasetsize
        self.checkpoint_path = checkpoint_path

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        ckpt_file_name = "faster_rcnn-{}_{}.ckpt".format(cur_epoch, self.datasetsize)
        checkpoint_path = os.path.join(self.checkpoint_path, ckpt_file_name)
        if cur_epoch % self.config.interval == 0 or cur_epoch == self.config.epoch_size:
            self.apply_eval(self.config, checkpoint_path)