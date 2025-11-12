import yaml
from argparse import ArgumentParser

from flashi2v.data.utils.wan_utils import WanVideoFilter
from flashi2v.data.utils.utils import LMDBReader, LMDBWriter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filter_config', type=str, default='configs/filter_config.json')
    args = parser.parse_args()
    filter_config = args.filter_config
    with open(filter_config, 'r', encoding='utf-8') as f:
        filter_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    video_filter =  WanVideoFilter(**filter_config)
    filtered_data_samples = video_filter.filter_data_samples()
    metafile_writer = LMDBWriter()
    metafile_writer.save_filtered_data_samples(filtered_data_samples, save_path=filter_config['save_path'])
    metafile_reader = LMDBReader(filter_config['save_path'])
    print(metafile_reader.getitem(0))
    print(metafile_reader.getitem(10000))

