import os
import re
import pickle
import lmdb
import zlib
import html
import ftfy
import json
import pandas as pd
import zstandard as zstd
from abc import ABC, abstractmethod
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict
import urllib.parse as ul
from bs4 import BeautifulSoup
from mindspeed_mm.utils.utils import Registry

def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def get_dataset_list(ann_path):
    # Loading annotations from files
    print(f"loading annotations from {ann_path} ...")
    if ann_path.endswith('.csv'):
        with open(ann_path, 'r') as csvfile:
            dataset = list(csv.DictReader(csvfile))
    elif ann_path.endswith('.json'):
        with open(ann_path, 'r') as f:
            dataset = json.load(f)
    else:
        raise NotImplementedError
    return dataset


def read_ann_txt(ann_txt_path, use_absolute_path=True):
    cap_lists = []
    with open(ann_txt_path, "r") as f:
        folder_anno = [
            i.strip().split(",")
            for i in f.readlines()
            if len(i.strip()) > 0
        ]
    for folder, anno in tqdm(folder_anno, desc="Loading annotations"):
        sub_list = get_dataset_list(anno)
        if use_absolute_path:
            for sub in sub_list:
                sub["path"] = os.path.join(folder, sub["path"])
        cap_lists += sub_list
    return cap_lists


class DataProcessor:
    """
    Factory class for creating data processor instances
    """
    @staticmethod
    def create(data_processor_type=None, *args, **kwargs) -> "AbstractDataProcessor":
        """
        Initialize with specified data processor type
        
        Args:
            data_processor_type: Registered data backend type
        """
        processor_cls = Registry.get_class(data_processor_type)
        return processor_cls(*args, **kwargs)

class DataFilter:
    """
    Factory class for creating data filter instances
    """
    @staticmethod
    def create(data_filter_type=None, *args, **kwargs) -> "AbstractDataFilter":
        """
        Initialize with specified data filter type
        
        Args:
            data_filter_type: Registered data backend type 
        """
        filter_cls = Registry.get_class(data_filter_type)
        return filter_cls(*args, **kwargs)

class MetaFileReader:
    """
    Factory class for creating metafile reader instances 
    """
    @staticmethod
    def create(metafile_reader_type=None, *args, **kwargs) -> "ParquetReader":
        """
        Initialize with specified metafile reader
        
        Args:
            metafile_reader_type: Registered metafile reader backend type 
        """
        reader_cls = Registry.get_class(metafile_reader_type)
        return reader_cls(*args, **kwargs)

class MetafileWriter:
    """
    Factory class for creating metafile writer instances 
    """
    @staticmethod
    def create(metafile_writer_type=None, *args, **kwargs) -> "ParquetWriter":
        """
        Initialize with specified metafile writer
        
        Args:
            metafile_writer_type: Registered metafile writer backend type 
        """
        writer_cls = Registry.get_class(metafile_writer_type)
        return writer_cls(*args, **kwargs)


class AbstractMetafileReader(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def getitem(self):
        pass

@Registry.register
class ParquetReader(AbstractMetafileReader):
    def __init__(self, parquet_path, columns=None, cache_row_group_size=1, **kwargs):
        """
        Reader for Parquet-format data with optional row-group caching.

        Args:
            parquet_path (str): Path to the Parquet file or directory.
            columns (list, optional): List of column names to load. Load all if None.
            cache_row_group_size (int): Number of rows to cache in memory. 
                                         If >1, reads are grouped to improve speed.
        """
        self.parquet_path = parquet_path
        self.columns = columns
        self.dataset = ds.dataset(parquet_path, format="parquet")
        self._num_rows = self.dataset.count_rows()

        self.cache_row_group_size = cache_row_group_size
        self.cached_row_group = None
        self.cached_start_index = None

    def __len__(self):
        return self._num_rows

    def getitem(self, index):
        """
        Retrieve a single sample by index, optionally using row-group caching.

        Args:
            index (int): Row index to read.

        Returns:
            dict: Sample data as a dictionary.
        """
        if self.cache_row_group_size > 1:
            if (self.cached_row_group is None or
                not (self.cached_start_index <= index < self.cached_start_index + len(self.cached_row_group))):
                start = (index // self.cache_row_group_size) * self.cache_row_group_size
                end = min(start + self.cache_row_group_size, self._num_rows)

                indices = pa.array(list(range(start, end)))
                self.cached_row_group = self.dataset.take(indices, columns=self.columns)
                self.cached_start_index = start

            offset = index - self.cached_start_index
            item = {k: v[offset] for k, v in self.cached_row_group.to_pydict().items()}
        else:
            indices = pa.array([index])
            row_dict = self.dataset.take(indices, columns=self.columns).to_pydict()
            item = {k: v[0] for k, v in row_dict.items()}
        return item

@Registry.register
class LMDBReader:
    def __init__(self, lmdb_path, decompression=False, readonly=True, lock=False, **kwargs):
        """
        Reader for LMDB-format data. Supports a single LMDB or a directory containing multiple *.lmdb subfolders.

        Args:
            lmdb_path (str): Path to a .lmdb folder or a directory containing multiple .lmdb folders.
            readonly (bool): Open the LMDB(s) in read-only mode.
            lock (bool): Whether to enable locking. Use False for faster reads.
        """
        self.decompression = decompression
        self.readonly = readonly
        self.lock = lock

        # Determine whether it's a single .lmdb or a folder of multiple .lmdbs
        if os.path.isdir(lmdb_path) and lmdb_path.endswith(".lmdb"):
            self.lmdb_dirs = [lmdb_path]
        elif os.path.isdir(lmdb_path):
            self.lmdb_dirs = sorted(
                [os.path.join(lmdb_path, d) for d in os.listdir(lmdb_path)
                 if os.path.isdir(os.path.join(lmdb_path, d)) and d.endswith(".lmdb")]
            )
            if not self.lmdb_dirs:
                raise ValueError(f"No .lmdb folders found in directory: {lmdb_path}")
        else:
            raise ValueError(f"Invalid LMDB path: {lmdb_path}")

        # Open all LMDB environments
        self.envs = []
        self.lengths = []
        self.offsets = []

        offset = 0
        for path in self.lmdb_dirs:
            env = lmdb.open(
                path,
                readonly=self.readonly,
                lock=self.lock,
                readahead=False,
                max_readers=2048,
                map_size=1 << 40,
            )
            with env.begin() as txn:
                n = txn.stat()["entries"]
            self.envs.append(env)
            self.lengths.append(n)
            self.offsets.append(offset)
            offset += n

        self._total = sum(self.lengths)

    def __len__(self):
        return self._total

    def decompress(self, data):
        if self.decompression:
            data = zstd.decompress(data)
        return data

    def getitem(self, index: int):
        """
        Retrieve a sample by global index from potentially multiple LMDB shards.

        Args:
            index (int): Global index across all LMDB files.

        Returns:
            object: Deserialized sample object.
        """
        if not (0 <= index < self._total):
            raise IndexError(f"Index out of bounds: {index}")

        # Find which env the index belongs to
        for i in range(len(self.envs)):
            start = self.offsets[i]
            end = start + self.lengths[i]
            if start <= index < end:
                local_index = index - start
                key = f"{local_index:012d}".encode("utf-8")
                with self.envs[i].begin() as txn:
                    value = txn.get(key)
                    if value is None:
                        raise KeyError(f"Key not found in LMDB shard {i}: {key}")
                    return pickle.loads(self.decompress(value))

        raise RuntimeError("Should not reach here")

    def close(self):
        """Close all LMDB environments."""
        for env in self.envs:
            if env is not None:
                env.close()
        self.envs = []


class AbstractDataProcessor(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_one_sample(self, path, meta_info=None):
        return sample

    @abstractmethod
    def process_one_sample(self, sample, *args, **kwargs):
        return processed_sample

class AbstractDataFilter(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter_data_samples(self, data_samples):
        '''
        Filter data samples

        Args:
            data_samples: list of data samples

        Returns:
            list of filtered data samples
        '''

        return filtered_data_samples


class AbstractMetafileWriter(ABC):

    save_type_checklist = ["parquet", "lmdb"]

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_filtered_data_samples(self, data_samples, save_path, save_name_extra_info, **kwargs):
        pass

    @abstractmethod
    def save_filtered_data_samples_for_multi_ranks(self, data_samples, save_path, num_ranks, save_name_extra_info, **kwargs):
        pass

    @abstractmethod
    def add_filtered_data_to_exist_metafile(self, data_samples, metafile_path, **kwargs):
        pass

    @abstractmethod
    def add_filtered_data_to_exist_metafile_for_multi_ranks(self, data_samples, save_dir, num_ranks, **kwargs):
        pass


    def get_real_save_path(self, save_path, total_num, extra_info=None, save_type="parquet"):
        """

        Args:
            save_path: output file path (e.g., 'output.parquet')
            total_num: total number of data samples
            extra_info: extra information to append to the file name (e.g., 'train')
        
        Returns:
            output file path (e.g., 'output.parquet')

        """

        prefix = f"filtered_samples_{total_num}"

        if extra_info is not None:
            prefix = f"{prefix}_{extra_info}"

        
        if save_path.endswith(".parquet") or save_path.endswith(".lmdb"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file_name = f"{prefix}_{os.path.basename(save_path)}"
            save_path = os.path.join(os.path.dirname(save_path), file_name)
            return save_path
        else:
            assert save_type in self.save_type_checklist, f"Invalid save type: {save_type}"
            try:
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f"{prefix}.{save_type}")
                return save_path
            except:
                raise ValueError(f"Invalid save path: {save_path}")

    def get_new_name_after_adding(self, file_path, added_num):
        assert file_path.split('.')[-1] in self.save_type_checklist, f"Invalid save type: {file_path.split('.')[-1]}"
        match = re.match(r'(filtered_samples_)(\d+)(_.*)', file_path)
        prefix = match.group(1)
        orig_num = int(match.group(2))
        postfix = match.group(3)
        new_num = orig_num + added_num
        save_path = f"{prefix}{new_num}{postfix}"
        return save_path

    
@Registry.register
class ParquetWriter(AbstractMetafileWriter):

    def _write_parquet_file(
        self,
        data_slice,
        output_path,
        schema,
        batch_size=8192,
    ):
        """
        Write a list of data samples to a Parquet file in batches.

        Args:
            data_slice: list of data samples to write (list[dict])
            output_path: full file path to save the Parquet file
            schema: pyarrow schema used for writing consistency
            batch_size: number of records per write batch

        Returns:
            None
        """

        writer = None
        total = len(data_slice)

        for i in tqdm(range(0, total, batch_size), desc="Writing parquet"):
            batch = data_slice[i: i + batch_size]
            table = pa.Table.from_pylist(batch, schema)

            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)

            writer.write_table(table)

        if writer:
            writer.close()

    def _append_parquet_file(
        self,
        data_slice,
        output_path,
        schema,
        batch_size=8192,
    ):
        """
        Append a list of data samples to an existing Parquet file, or create it if not found.

        Args:
            data_slice: list of data samples to append (list[dict])
            output_path: full path to the target Parquet file
            schema: pyarrow schema for compatibility check
            batch_size: number of records per append batch

        Returns:
            None
        """

        writer = None
        if not os.path.exists(output_path):
            writer = pq.ParquetWriter(output_path, schema)
        else:
            existing_schema = pq.read_schema(output_path)
            if existing_schema != schema:
                raise ValueError(f"Schema mismatch when appending to existing file: {output_path}")
            writer = pq.ParquetWriter(output_path, schema)

        total = len(data_slice)
        for i in tqdm(range(0, total, batch_size), desc="Appending parquet"):
            batch = data_slice[i: i + batch_size]
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table)

        if writer:
            writer.close()


    def save_filtered_data_samples(
        self,
        data_samples,
        save_path,
        save_batch_size=8192,
        save_name_extra_info=None,
    ):
        """
        Save filtered data samples to a single Parquet file with batched writing.

        Args:
            data_samples: list of data samples to save (list[dict])
            save_path: output file path (e.g., 'output.parquet')
            save_batch_size: number of records per write batch

        Returns:
            None
        """
        total = len(data_samples)

        save_path = self.get_real_save_path(save_path, total, extra_info=save_name_extra_info, save_type="parquet")
        
        schema = pa.Table.from_pylist(data_samples[:min(1000, total)]).schema

        print(f"[INFO] Saving {total} samples to '{save_path}' as Parquet...")

        self._write_parquet_file(data_samples, save_path, schema, save_batch_size)

        print(f"[INFO] Done. File saved to '{save_path}'.")


    def save_filtered_data_samples_for_multi_ranks(
        self,
        data_samples,
        save_dir,
        num_ranks=1,
        file_prefix="rank",
        num_threads=4,
        save_batch_size=8192,
        save_name_extra_info=None,
    ):
        """
        Save filtered data samples into multiple Parquet files (one per rank), in parallel.

        Args:
            data_samples: list of data samples to save (list[dict])
            save_dir: target directory to save the Parquet files
            num_ranks: number of shards (typically one per distributed training rank)
            file_prefix: prefix used for each output file (e.g., 'rank_00.parquet')
            num_threads: number of threads for concurrent writing
            save_batch_size: number of records per write batch

        Returns:
            None
        """


        os.makedirs(save_dir, exist_ok=True)
        total = len(data_samples)
        per_rank = total // num_ranks
        print(f"[INFO] Saving {total} samples to {num_ranks} files in '{save_dir}' using {num_threads} threads...")

        schema = pa.Table.from_pylist(data_samples[:min(1000, total)]).schema 

        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for rank in tqdm(range(num_ranks), desc="Saving shards"):
                print(f"[INFO] Saving shard {rank:02d}...")
                start = rank * per_rank
                end = (rank + 1) * per_rank if rank < num_ranks - 1 else total
                part_data = data_samples[start:end]
                extra = f"rank_{rank_id}_{save_name_extra_info}" if save_name_extra_info else f"rank_{rank_id}"
                save_path = self.get_real_save_path(save_dir, per_rank, extra_info=save_name_extra_info, save_type="parquet")
                futures.append(executor.submit(self._write_parquet_file, part_data, file_path, schema, save_batch_size))

        print(f"[INFO] Finished saving {num_ranks} shards to '{save_dir}'.")

    def add_filtered_data_to_exist_metafile(
        self,
        data_samples,
        metafile_path,
        save_batch_size=8192
    ):
        """
        Append filtered data samples to an existing single Parquet file.

        Args:
            data_samples: list of new data samples to append (list[dict])
            metafile_path: path to the existing Parquet file
            save_batch_size: number of records per append batch

        Returns:
            None
        """

        added_num = len(data_samples)

        if not os.path.exists(metafile_path):
            raise FileNotFoundError(f"Target metafile does not exist: {metafile_path}")

        schema = pa.read_schema(metafile_path)
        print(f"[INFO] Appending {len(data_samples)} samples to existing file '{metafile_path}'...")
        self._append_parquet_file(data_samples, metafile_path, schema, batch_size=save_batch_size)
        print(f"[INFO] Done appending to '{metafile_path}'.")
        new_save_path = self.get_new_name_after_adding(metafile_path, added_num)
        os.rename(metafile_path, new_save_path)
        print(f"[INFO] Renamed '{metafile_path}' to '{new_save_path}'.")


    def add_filtered_data_to_exist_metafile_for_multi_ranks(
        self,
        data_samples,
        save_dir,
        num_ranks=1,
        file_prefix="rank",
        num_threads=4,
        save_batch_size=8192
    ):
        """
        Append filtered data samples to multiple existing Parquet files (one per rank), in parallel.

        Args:
            data_samples: list of data samples to append (list[dict])
            save_dir: directory containing existing shard files
            num_ranks: number of shards (typically one per distributed training rank)
            file_prefix: prefix used to identify shard files
            num_threads: number of threads for concurrent appending
            save_batch_size: number of records per append batch

        Returns:
            None
        """

        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")

        total = len(data_samples)
        per_rank = total // num_ranks
        print(f"[INFO] Appending {total} samples to {num_ranks} files in '{save_dir}' using {num_threads} threads...")

        meta_file_path_list = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir)])
        schema = pq.read_schema(meta_file_path_list[0])

        def append_rank(rank_id: int):
            start = rank_id * per_rank
            end = (rank_id + 1) * per_rank if rank_id < num_ranks - 1 else total
            part_data = data_samples[start:end]
            file_path = meta_file_path_list[rank_id]
            print(f"[INFO] Appending shard {rank_id}...")
            self._append_parquet_file(part_data, file_path, schema, batch_size=save_batch_size)
            new_save_path = self.get_new_name_after_adding(file_path, end - start)
            os.rename(file_path, new_save_path)
            print(f"[INFO] Renamed '{file_path}' to '{new_save_path}'.")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(append_rank, range(num_ranks)), total=num_ranks, desc="Appending shards"))

        print(f"[INFO] Finished appending to {num_ranks} shards in '{save_dir}'.")


@Registry.register
class LMDBWriter(AbstractMetafileWriter):
    def __init__(self, map_size=1 << 40, compression=False, **kwargs):  # 默认 1 TB 空间
        self.map_size = map_size
        self.compression = compression

    def compress(self, data):
        if self.compression:
            data = zstd.compress(data)
        return data

    def _write_lmdb_file(self, data_slice, output_path):
        """
        Write a list of data samples to an LMDB file.

        Args:
            data_slice: list of data samples to write (list[dict or any pickle-serializable object])
            output_path: full file path to save the LMDB database
        """
        print(f"[INFO] Writing {len(data_slice)} samples to '{output_path}' as LMDB...")
        env = lmdb.open(output_path, map_size=self.map_size)
        with env.begin(write=True) as txn:
            for i, sample in enumerate(tqdm(data_slice, desc="Writing LMDB")):
                key = f"{i:012d}".encode("utf-8")
                value = pickle.dumps(sample)
                value = self.compress(value)
                txn.put(key, value)
        env.close()
        print(f"[INFO] Done writing to '{output_path}'.")

    def _append_lmdb_file(self, data_slice, output_path):
        """
        Append a list of data samples to an existing LMDB file.

        Args:
            data_slice: list of new data samples to append
            output_path: existing LMDB file path
        """
        print(f"[INFO] Appending {len(data_slice)} samples to existing LMDB file '{output_path}'...")
        env = lmdb.open(output_path, map_size=self.map_size)
        with env.begin(write=False) as txn:
            stat = txn.stat()
            last_id = stat["entries"]
        with env.begin(write=True) as txn:
            for i, sample in enumerate(tqdm(data_slice, desc="Appending LMDB")):
                key = f"{last_id + i:012d}".encode("utf-8")
                value = pickle.dumps(sample)
                value = self.compress(value)
                txn.put(key, value)
        env.close()
        print(f"[INFO] Done appending to '{output_path}'.")

    def save_filtered_data_samples(self, data_samples, save_path, save_name_extra_info=None):
        """
        Save filtered data samples to a single LMDB file.

        Args:
            data_samples: list of data samples to save
            save_path: target file path or directory
            save_name_extra_info: optional string added to file name
        """
        total_num = len(data_samples)
        save_path = self.get_real_save_path(save_path, total_num, extra_info=save_name_extra_info, save_type="lmdb")
        self._write_lmdb_file(data_samples, save_path)

    def save_filtered_data_samples_for_multi_ranks(
        self,
        data_samples,
        save_dir,
        num_ranks=1,
        file_prefix="rank",
        num_threads=4,
        save_name_extra_info=None,
    ):
        """
        Save filtered data samples into multiple LMDB files (one per rank), in parallel.

        Args:
            data_samples: list of data samples to save
            save_dir: target directory to save the LMDB files
            num_ranks: number of shards (typically one per distributed training rank)
            file_prefix: file name prefix for each rank (e.g., rank_00.lmdb)
            num_threads: number of concurrent threads
        """
        os.makedirs(save_dir, exist_ok=True)
        total = len(data_samples)
        per_rank = total // num_ranks

        print(f"[INFO] Saving {total} samples to {num_ranks} LMDB files in '{save_dir}' using {num_threads} threads...")

        def save_rank(rank_id):
            start = rank_id * per_rank
            end = (rank_id + 1) * per_rank if rank_id < num_ranks - 1 else total
            part_data = data_samples[start:end]
            extra = f"rank_{rank_id}_{save_name_extra_info}" if save_name_extra_info else f"rank_{rank_id}"
            save_path = self.get_real_save_path(save_dir, len(part_data), extra_info=extra, save_type="lmdb")
            print(f"[INFO] Saving shard {rank_id} to '{save_path}'...")
            self._write_lmdb_file(part_data, save_path)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(save_rank, range(num_ranks)), total=num_ranks, desc="Saving LMDB shards"))

        print(f"[INFO] Finished saving {num_ranks} LMDB shards to '{save_dir}'.")

    def add_filtered_data_to_exist_metafile(self, data_samples, metafile_path):
        """
        Append filtered data samples to an existing single LMDB file.

        Args:
            data_samples: list of new data samples to append
            metafile_path: path to the existing LMDB file
        """
        added_num = len(data_samples)
        if not os.path.exists(metafile_path):
            raise FileNotFoundError(f"Target metafile does not exist: {metafile_path}")

        self._append_lmdb_file(data_samples, metafile_path)

        new_save_path = self.get_new_name_after_adding(metafile_path, added_num)
        os.rename(metafile_path, new_save_path)
        print(f"[INFO] Renamed '{metafile_path}' to '{new_save_path}' after appending.")

    def add_filtered_data_to_exist_metafile_for_multi_ranks(
        self,
        data_samples,
        save_dir,
        num_ranks=1,
        file_prefix="rank",
        num_threads=4,
    ):
        """
        Append filtered data samples to multiple existing LMDB files (one per rank), in parallel.

        Args:
            data_samples: list of data samples to append
            save_dir: directory containing existing shard files
            num_ranks: number of shards
            file_prefix: prefix used to identify shard files
            num_threads: number of concurrent threads
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")

        meta_file_path_list = sorted([
            os.path.join(save_dir, f) for f in os.listdir(save_dir)
            if f.endswith(".lmdb")
        ])

        total = len(data_samples)
        per_rank = total // num_ranks

        print(f"[INFO] Appending {total} samples to {num_ranks} LMDB files in '{save_dir}' using {num_threads} threads...")

        def append_rank(rank_id):
            start = rank_id * per_rank
            end = (rank_id + 1) * per_rank if rank_id < num_ranks - 1 else total
            part_data = data_samples[start:end]
            file_path = meta_file_path_list[rank_id]
            self._append_lmdb_file(part_data, file_path)
            new_save_path = self.get_new_name_after_adding(file_path, end - start)
            os.rename(file_path, new_save_path)
            print(f"[INFO] Renamed '{file_path}' to '{new_save_path}' after appending shard {rank_id}.")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(append_rank, range(num_ranks)), total=num_ranks, desc="Appending LMDB shards"))

        print(f"[INFO] Finished appending to {num_ranks} LMDB shards in '{save_dir}'.")

class AbstractTextProcessor(ABC):
    
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_text(self, text):
        return text

@Registry.register   
class TextProcessor(AbstractTextProcessor):
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + "\)"
        + "\("
        + "\]"
        + "\["
        + "\}"
        + "\{"
        + "\|"
        + "\\"
        + "\/"
        + "\*"
        + r"]{1,}"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_text(self, text):
        return self.text_preprocessing(text)

    @staticmethod
    def text_preprocessing(text, use_clean_caption=True, support_chinese=False):
        if use_clean_caption:
            text = TextProcessor.clean_caption(text, support_chinese=support_chinese)
            text = TextProcessor.clean_caption(text, support_chinese=support_chinese)
        else:
            text = text.lower().strip()
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @staticmethod
    def clean_caption(caption, support_chinese=False):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        if not support_chinese:
            caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # Uniform quotation marks
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            TextProcessor.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = TextProcessor.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()