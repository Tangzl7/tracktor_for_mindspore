import os
import time
import tqdm
import yaml
import numpy as np
import os.path as osp

from src.tracktor.tracker import Tracker
from src.tracktor.reid import ResNet50_FC512
from src.tracktor.datasets.factory import Datasets
from src.tracktor.utils import get_mot_accum, \
                    plot_sequence, evaluate_mot_accums
from src.tracktor.frcnn_fpn import FRCNN_FPN, FRCNN_FPN_Fea
from src.tracktor.config import config as tracktor_config
from src.frcnn.model_utils.config import config as frcnn_config

import mindspore.common.dtype as mstype
from mindspore import Parameter, context
from mindspore import load_checkpoint, load_param_into_net

context.set_context(mode=context.PYNATIVE_MODE, device_target=tracktor_config.device_target, device_id=tracktor_config.device_id)


def get_weights(obj_detect_models, dataset):
    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)

    return obj_detect_models, dataset


if __name__ == '__main__':
    output_dir = tracktor_config.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = Datasets(tracktor_config.dataset)
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"

    ##########################
    # Initialize the modules #
    ##########################

    print("Initializing object detector(s).")
    obj_detect = FRCNN_FPN(frcnn_config)
    obj_detect_fea = FRCNN_FPN_Fea(frcnn_config)
    param_dict = load_checkpoint(tracktor_config.obj_detect_weight)
    if tracktor_config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    load_param_into_net(obj_detect, param_dict)
    load_param_into_net(obj_detect_fea, param_dict)
    obj_detect.set_train(False)
    obj_detect_fea.set_train(False)

    print("Initializing reid.")
    reid = ResNet50_FC512()
    param_dict_reid = load_checkpoint(tracktor_config.reid_weight)
    if tracktor_config.device_target == "GPU":
        for key, value in param_dict_reid.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict_reid[key] = Parameter(tensor, key)
    load_param_into_net(reid, param_dict_reid)
    reid.set_train(False)

    if device_type == "Ascend":
        print('device type: ascend')
        obj_detect.to_float(mstype.float16)
        obj_detect_fea.to_float(mstype.float16)
        reid.to_float(mstype.float16)

    tracker = Tracker(obj_detect, obj_detect_fea, reid, tracker_cfg=tracktor_config.tracker)
    time_total, num_frames, mot_accums = 0, 0, []

    for seq in dataset:
        tracker.obj_detect = obj_detect
        tracker.reset()

        print(f"Tracking: {seq}")

        num_frames += len(seq)

        results = {}
        if tracktor_config.load_results:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()
            for frame_data in tqdm.tqdm(seq):
                tracker.step(frame_data)

            results = tracker.get_result()
            time_total += time.time() - start

            print(f"Tracks found: {len(results)}")
            print(f"Runtime for {seq}: {time.time() - start :.2f} s.")
            print(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        if seq.no_gt:
            print("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        if tracktor_config.write_images:
            plot_sequence(results, seq, osp.join(output_dir,
                                str(dataset), str(seq)), tracktor_config.write_images)
    if time_total:
        print(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        print("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)

