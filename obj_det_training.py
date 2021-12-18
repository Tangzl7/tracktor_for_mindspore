import argparse
import os.path as osp

from src.frcnn.model_utils.config import config
from src.frcnn.lr_schedule import dynamic_lr_1
from src.frcnn.mot_data import preprocess_fn
from src.frcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.frcnn.mot_data import MOTObjDetectDatasetGenerator
from src.frcnn.model_utils.moxing_adapter import moxing_wrapper
from src.frcnn.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.frcnn.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

from mindspore.nn import SGD
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor

set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

if config.run_distribute:
    if config.device_target == "Ascend":
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
else:
    rank = 0
    device_num = 1


def get_dataset():
    train_data_dir = config.train_data
    if config.enable_modelarts:
        train_data_dir = config.data_path + '/MOT17/train'
    train_split_seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
    dataset_generator = MOTObjDetectDatasetGenerator(root=train_data_dir, split_seqs=train_split_seqs)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'img_shape', 'boxes', 'labels', 'valid_num', 'image_id'],
                                  shuffle=True, python_multiprocessing=config.python_multiprocessing, num_shards=device_num, shard_id=rank)
    preprocess_func = (lambda img, img_shape, boxes, labels, valid_num, image_id:
                        preprocess_fn(img, img_shape, boxes, labels, valid_num, image_id, 0.5))
    dataset = dataset.map(input_columns=['img', 'img_shape', 'boxes', 'labels', 'valid_num', 'image_id'],
                        output_columns=['img', 'img_shape', 'boxes', 'labels', 'valid_num'],
                        column_order=['img', 'img_shape', 'boxes', 'labels', 'valid_num'],
                        operations=preprocess_func)
    dataset = dataset.batch(batch_size=config.batch_size, drop_remainder=True)
    return dataset_generator, dataset


def get_detection_model():
    config.num_classes = 2
    model = Faster_Rcnn_Resnet50(config)
    model.set_train(True)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        model.to_float(mstype.float16)

    return model

def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path
    config.train_data = config.data_path + '/MOT17/train'
    config.pre_trained = config.output_path + '/faster_rcnn_fpn/pretraining/fasterrcnn_resnet50_fpn.ckpt'


@moxing_wrapper(pre_process=modelarts_pre_process)
def train(model):
    _, dataset = get_dataset()
    print("train start...")
    loss = LossNet()
    lr = Tensor(dynamic_lr_1(config, dataset.get_dataset_size()), mstype.float32)

    opt = SGD(params=model.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    load_path = config.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn" in item:
                param_dict.pop(item)
        load_param_into_net(model, param_dict)
        print("load model success...")

    model_with_loss = WithLossCell(model, loss)
    if config.run_distribute:
        model = TrainOneStepCell(model_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                                 mean=True, degree=device_num)
    else:
        model = TrainOneStepCell(model_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint and rank == 0:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset.get_dataset_size(),
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = osp.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    if config.run_eval and rank == 0:
        from src.frcnn.eval_callback import EvalCallBack
        from src.frcnn.eval_utils import apply_eval
        save_checkpoint_path = osp.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        eval_cb = EvalCallBack(config, apply_eval, dataset.get_dataset_size(), save_checkpoint_path)
        cb += [eval_cb]

    model = Model(model)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == '__main__':
    model = get_detection_model()
    train(model)
