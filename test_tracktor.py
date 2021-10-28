import os
import yaml
import os.path as osp
from sacred import Experiment

from src.tracktor.config import get_output_dir

from mindspore import load_checkpoint, load_param_into_net

ex = Experiment()
ex.add_config('./tracktor.yaml')

webcan = 'data/traffic_short.mp4'

@ex.automain
def main(tracktor, _config):
    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("[*] Building object detector")
    if tracktor['network'] == 'frcnn':
        # FRCNN
        from src.tracktor.frcnn_fpn import FRCNN_FPN
        from src.frcnn.config import config
        obj_detect = FRCNN_FPN(config)
        """
        if osp.exists(tracktor['obj_detect_weights']):
            param_dict = load_checkpoint(tracktor['obj_detect_weights'])
            load_param_into_net(obj_detect, param_dict)
        else:
            raise FileNotFoundError(f"detector's weight file is not exist")
        """
    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")