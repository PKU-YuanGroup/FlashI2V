import yaml
from argparse import ArgumentParser

import os
import glob
import importlib.util
import mindspeed.megatron_adaptor

from mindspeed_mm.data.data_utils.utils import DataFilter, MetaFileReader, MetafileWriter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filter_config', type=str, default='configs/filter_config.json')
    parser.add_argument('--model', type=str, default="wan")
    args = parser.parse_args()
    filter_config = args.filter_config
    with open(filter_config, 'r', encoding='utf-8') as f:
        filter_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.model == "wan":
        video_filter = DataFilter.create('WanVideoFilter', **filter_config)
    filtered_data_samples = video_filter.filter_data_samples()
    metafile_writer = MetafileWriter.create('LMDBWriter')
    metafile_writer.save_filtered_data_samples(filtered_data_samples, save_path=filter_config['save_path'])
    metafile_reader = MetaFileReader.create('LMDBReader', filter_config['save_path'])
    print(metafile_reader.getitem(0))
    print(metafile_reader.getitem(10000))

