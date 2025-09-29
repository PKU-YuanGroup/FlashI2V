import os
import json
import yaml

from mindspeed_mm.utils.utils import get_dtype


class ConfigReader:
    """  
    read_config read json or yaml file dict processed by MMconfig
    and convert to class attributes, besides, read_config
    support to convert dict for specific purposes.
    """
    def __init__(self, config_dict: dict) -> None:
        for k, v in config_dict.items():
            if k == "dtype":
                v = get_dtype(v)
            if isinstance(v, dict):
                self.__dict__[k] = ConfigReader(v)
            else:
                self.__dict__[k] = v
    
    def to_dict(self) -> dict:
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret
    
    def __repr__(self) -> str:
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                print(">>>>> {}".format(k))
                print(v)
            else:
                print("{}: {}".format(k, v))
        return ""

    def __str__(self) -> str:
        try:
            self.__repr__()
        except Exception as e:
            print(f"An error occurred: {e}")
        return ""

    def update_unuse(self, **kwargs):

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


class MMConfig:
    """ 
    MMconfig 
        input: a dict of json or yaml path
    """
    def __init__(self, config_files: dict) -> None:
        for config_name, config_file_path in config_files.items():
            if os.path.exists(config_file_path):
                real_path = os.path.realpath(config_file_path)
                if real_path.endswith(".json"):
                    config_dict = self.read_json(real_path)
                elif real_path.endswith(".yaml"):
                    config_dict = self.read_yaml(real_path)
                setattr(self, config_name, ConfigReader(config_dict))
    
    @staticmethod
    def read_json(json_path):
        with open(json_path, mode="r") as f:
            json_file = f.read()
        config_dict = json.loads(json_file)
        return config_dict

    @staticmethod
    def read_yaml(yaml_path):
        with open(yaml_path, mode="r") as f:
            yaml_file = f.read()
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return config_dict
   
    
def _add_mm_args(parser):
    group = parser.add_argument_group(title="multimodel")
    group.add_argument("--clip_grad_ema_decay", type=float, default=0.99, help="EMA decay coefficient of Adaptive Gradient clipping in Open-Sora Plan based on global L2 norm.")
    group.add_argument("--mm-data", type=str, default="")
    group.add_argument("--mm-model", type=str, default="")
    group.add_argument("--mm-tool", type=str, default="")
    return parser


def mm_extra_args_provider(parser):
    parser = _add_mm_args(parser)
    return parser


def merge_mm_args(args):
    if not hasattr(args, "mm"):
        setattr(args, "mm", object)
        config_files = {"model": args.mm_model, "data": args.mm_data, "tool": args.mm_tool}
        args.mm = MMConfig(config_files)

