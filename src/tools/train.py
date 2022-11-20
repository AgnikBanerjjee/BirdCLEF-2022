"""
train model
Usage:
    train.py [--o=<path>]  [--cfg=<path>]
    train.py -h | --help
Options:
    -h --help           show this screen help
    -o=<path>           output path
    --cfg=<path>        config path 
"""


from docopt import docopt
from src.modeling.meta_arch.build import build_model
from data.birdclef_data import build_data_loader
from src.modeling.solver import (
    build_optimizer,
    build_scheduler,
    build_evaluator,
)

from src.config import get_cfg_defaults

# For Tensorboard integration
from torch.utils.tensorboard import SummaryWriter


def train(cfg, debug=False):
    pass


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments["-o"]
    cfg_path = arguments["--cfg"]
    cfg = get_cfg_defaults()
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    cfg.OUTPUT_PATH = output_path
    train(cfg)
