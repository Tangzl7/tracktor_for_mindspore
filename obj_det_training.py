import argparse
import numpy as np
import os.path as osp

from src.frcnn.config import config
from src.frcnn.lr_schedule import dynamic_lr
from src.tracktor.mot_data import preprocess_fn
from src.frcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.tracktor.mot_data import MOTObjDetectDatasetGenerator
from src.frcnn.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet

from mindspore.nn import SGD
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore import context, Parameter, Tensor
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor

set_seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--train_mot_dir', default='MOT17Det')
parser.add_argument('--test_mot_dir', default='MOT17Det')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--eval_train', action='store_true')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr_drop', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--train_vis_threshold', type=float, default=0.25)
parser.add_argument('--test_vis_threshold', type=float, default=0.25)
parser.add_argument('--pretraining', default='')
parser.add_argument('--arch', type=str, default='fasterrcnn_resnet50_fpn')

parser.add_argument("--device_target", type=str, default="CPU",
                    help="device where the code will be implemented, default is Ascend")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


def get_dataset():
    train_data_dir = osp.join(f'data/{args.train_mot_dir}', 'train')
    # test_data_dir = osp.join(f'data/{args.test_mot_dir}', 'train')
    train_split_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    # test_split_seqs = ['MOT17-09']
    dataset_generator = MOTObjDetectDatasetGenerator(root=train_data_dir, split_seqs=train_split_seqs)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'img_shape', 'boxes', 'labels', 'valid_num'], shuffle=True)
    dataset = dataset.map(input_columns=['img'], operations=py_vision.ToTensor())
    preprocess_func = (lambda img, boxes, labels, valid_num: preprocess_fn(img, boxes, labels, valid_num, 0.5))
    dataset = dataset.map(input_columns=['img', 'boxes', 'labels', 'valid_num'], operations=preprocess_func)
    dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    return dataset


def get_detection_model():
    config.num_classes = 2
    model = Faster_Rcnn_Resnet50(config)
    model.set_train()

    load_path = args.pretraining
    if load_path != "":
        param_dict = load_checkpoint(load_path)

        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }
        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                oldkey = newkey
            for k, v in key_mapping.items():
                if k in oldkey:
                    newkey = oldkey.replace(k, v)
                    param_dict[newkey] = param_dict.pop(oldkey)
                    break

        for item in list(param_dict.keys()):
            if not item.startswith('backbone'):
                param_dict.pop(item)

        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
        load_param_into_net(model, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        model.to_float(mstype.float16)

    return model


def train(model, dataset):
    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset.get_dataset_size()), mstype.float32)

    opt = SGD(params=model.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    model_with_loss = WithLossCell(model, loss)
    model = TrainOneStepCell(model_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossCallBack(rank_id=args.rank_id)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset.get_dataset_size(),
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = osp.join(config.save_checkpoint_path, "ckpt_" + str(args.rank_id) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(model)
    model.train(config.epoch_size, dataset, callbacks=cb)


if __name__ == '__main__':
    dataset = get_dataset()
    model = get_detection_model()
    train(model, dataset)