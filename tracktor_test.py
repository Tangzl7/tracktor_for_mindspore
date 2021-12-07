import os
import time
import tqdm
import yaml
import numpy as np
import os.path as osp
from sacred import Experiment

from src.frcnn.model_utils.config import config
from src.tracktor.tracker import Tracker
from src.tracktor.reid import ResNet50_FC512
from src.tracktor.config import get_output_dir
from src.tracktor.datasets.factory import Datasets
from src.tracktor.utils import get_mot_accum, \
                    plot_sequence, evaluate_mot_accums
from src.tracktor.frcnn_fpn import FRCNN_FPN, FRCNN_FPN_Fea

import mindspore.common.dtype as mstype
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
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"

    ##########################
    # Initialize the modules #
    ##########################

    _log.info("Initializing object detector(s).")
    obj_detect = FRCNN_FPN(config)
    obj_detect_fea = FRCNN_FPN_Fea(config)
    param_dict = load_checkpoint(tracktor['obj_detect_weight'])
    if tracktor['device_target'] == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    load_param_into_net(obj_detect, param_dict)
    load_param_into_net(obj_detect_fea, param_dict)
    obj_detect.set_train(False)
    obj_detect_fea.set_train(False)

    _log.info("Initializing reid.")
    reid = ResNet50_FC512()
    param_dict_reid = load_checkpoint(tracktor['reid_weight'])
    if tracktor['device_target'] == "GPU":
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

    tracker = Tracker(obj_detect, obj_detect_fea, reid, tracker_cfg=tracktor['tracker'])
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

