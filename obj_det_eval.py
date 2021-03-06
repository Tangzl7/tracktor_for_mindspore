import time
import argparse
import numpy as np
import os.path as osp

from src.frcnn.model_utils.config import config
from src.frcnn.util import bbox2result_1image
from src.frcnn.mot_data import preprocess_fn
from src.frcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.frcnn.mot_data import MOTObjDetectDatasetGenerator

import mindspore.dataset as ds
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore import context, Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net

set_seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--test_mot_dir', default='MOT17Det')
parser.add_argument('--pretraining', default='./ckpt/pretraining/model_epoch_27.ckpt')

parser.add_argument("--device_target", type=str, default="GPU",
                    help="device where the code will be implemented, default is Ascend")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 2.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


def get_dataset():
    test_data_dir = osp.join(f'./data/{args.test_mot_dir}', 'train')
    test_split_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    dataset_generator = MOTObjDetectDatasetGenerator(root=test_data_dir, split_seqs=test_split_seqs, train=False)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'img_shape', 'boxes', 'labels', 'valid_num', 'image_id'],
                                  shuffle=False)
    preprocess_func = (lambda img, img_shape, boxes, labels, valid_num, image_id:
                    preprocess_fn(img, img_shape, boxes, labels, valid_num, image_id, -1))
    dataset = dataset.map(input_columns=['img', 'img_shape', 'boxes', 'labels', 'valid_num', 'image_id'],
                        operations=preprocess_func)
    dataset = dataset.batch(batch_size=config.test_batch_size, drop_remainder=True)
    return dataset_generator, dataset


def get_detection_model():
    model = Faster_Rcnn_Resnet50(config)
    model.set_train(False)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        model.to_float(mstype.float16)

    return model


def evaluate_and_write_result_files(model, dataset, generator):
    param_dict = load_checkpoint(args.pretraining)
    if args.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    load_param_into_net(model, param_dict)

    results = {}
    eval_iter = 0
    total = dataset.get_dataset_size()

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 64

    for data in dataset.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1

        img_data = data['img']
        img_metas = data['img_shape']
        gt_bboxes = data['boxes']
        gt_labels = data['labels']
        gt_num = data['valid_num']
        img_id = data['image_id']

        start = time.time()
        # run net
        output = model(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
            results[str(img_id[j][0])] = {'boxes': outputs_tmp[0][:, :-1], 'scores': outputs_tmp[0][:, -1]}

    output_dir = './output/tracktor/MOT17/frcnn'
    generator.write_results_files(results, output_dir)
    generator.print_eval(results)


if __name__ == '__main__':
    generator, dataset = get_dataset()
    model = get_detection_model()
    evaluate_and_write_result_files(model, dataset, generator)
