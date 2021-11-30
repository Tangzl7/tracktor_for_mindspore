import os
import time
import cv2
import tqdm
import yaml
import numpy as np
import os.path as osp
import motmetrics as mm
from sacred import Experiment

from src.frcnn.config import config
from src.tracktor import data_handle
from src.tracktor.tracker import Tracker
from src.tracktor.frcnn_fpn import FRCNN_FPN
from src.tracktor.reid import ResNet50_FC512
from src.tracktor.config import get_output_dir
from src.tracktor.datasets.factory import Datasets
from src.tracktor.utils import get_mot_accum, \
                    plot_sequence, evaluate_mot_accums

import mindspore.dataset as ds
from mindspore import Parameter, context
from mindspore import load_checkpoint, load_param_into_net

ex = Experiment()
ex.add_config('./tracktor.yaml')

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

def get_weights(obj_detect_models, dataset):
    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)

    return obj_detect_models, dataset


@ex.automain
def main(tracktor, _config, _log):
    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    dataset = Datasets(tracktor['dataset'])

    ##########################
    # Initialize the modules #
    ##########################

    _log.info("Initializing object detector(s).")
    obj_detect = FRCNN_FPN(config)
    param_dict = load_checkpoint(tracktor['obj_detect_weight'])
    if tracktor['device_target'] == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    load_param_into_net(obj_detect, param_dict)
    obj_detect.set_train(False)

    _log.info("Initializing reid.")
    reid = ResNet50_FC512()
    param_dict_reid = load_checkpoint(tracktor['reid_weight'])
    if tracktor['device_target'] == "GPU":
        for key, value in param_dict_reid.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict_reid[key] = Parameter(tensor, key)
    load_param_into_net(reid, param_dict_reid)
    reid.set_train(False)

    tracker = Tracker(obj_detect, reid, tracker_cfg=tracktor['tracker'])
    time_total, num_frames, mot_accums = 0, 0, []

    for seq in dataset:
        tracker.obj_detect = obj_detect
        tracker.reset()

        _log.info(f"Tracking: {seq}")

        num_frames += len(seq)

        results = {}
        if tracktor['load_results']:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()
            for frame_data in tqdm.tqdm(seq):
                tracker.step(frame_data)

            results = tracker.get_result()
            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")
            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        if seq.no_gt:
            _log.info("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        if tracktor['write_images']:
            plot_sequence(results, seq, osp.join(output_dir,
                                str(dataset), str(seq)), tracktor['write_images'])
    if time_total:
        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        _log.info("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)

