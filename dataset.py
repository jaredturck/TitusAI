import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_utils import load_manifest


class TitusDataset(Dataset):
    def __init__(self, dataset_path, max_open_shards=8):
        self.dataset_path = Path(dataset_path)
        self.manifest = load_manifest(self.dataset_path)
        self.sequence_length = self.manifest['sequence_length']
        self.stored_sequence_length = self.manifest['stored_sequence_length']
        self.shards = self.manifest['shards']
        self.cumulative_sequences = []
        self.opened_shards = OrderedDict()
        self.max_open_shards = max_open_shards

        source_names = sorted({shard['source'] for shard in self.shards})
        self.source_ids = {
            source_name: source_id
            for source_id, source_name in enumerate(source_names)
        }

        total = 0
        for shard in self.shards:
            total += shard['num_sequences']
            self.cumulative_sequences.append(total)

        self.num_sequences = total

    def __len__(self):
        return self.num_sequences

    def __getstate__(self):
        state = dict(self.__dict__)
        state['opened_shards'] = OrderedDict()
        return state

    def shard_start(self, shard_index):
        if shard_index == 0:
            return 0
        return self.cumulative_sequences[shard_index - 1]

    def locate_sequence(self, idx):
        if idx < 0:
            idx += self.num_sequences

        assert 0 <= idx < self.num_sequences
        shard_index = bisect.bisect_right(self.cumulative_sequences, idx)
        return shard_index, idx - self.shard_start(shard_index)

    def close_shard(self, opened):
        for array in opened:
            if hasattr(array, '_mmap') and array._mmap is not None:
                array._mmap.close()

    def open_shard(self, shard_index):
        if shard_index in self.opened_shards:
            opened = self.opened_shards.pop(shard_index)
            self.opened_shards[shard_index] = opened
            return opened

        shard = self.shards[shard_index]
        shape = (shard['num_sequences'], self.stored_sequence_length)
        token_array = np.memmap(
            self.dataset_path / shard['tokens'],
            mode='r',
            dtype=np.uint16,
            shape=shape,
        )
        segment_array = np.memmap(
            self.dataset_path / shard['segments'],
            mode='r',
            dtype=np.uint16,
            shape=shape,
        )
        loss_mask_array = None
        if shard.get('loss_mask') is not None:
            loss_mask_array = np.memmap(
                self.dataset_path / shard['loss_mask'],
                mode='r',
                dtype=np.uint8,
                shape=shape,
            )

        opened = (token_array, segment_array, loss_mask_array)
        self.opened_shards[shard_index] = opened

        if len(self.opened_shards) > self.max_open_shards:
            _, old_opened = self.opened_shards.popitem(last=False)
            self.close_shard(old_opened)

        return opened

    def __getitem__(self, idx):
        shard_index, local_index = self.locate_sequence(idx)
        tokens, segments, loss_mask = self.open_shard(shard_index)

        token_ids = np.array(tokens[local_index], dtype=np.int64, copy=True)
        segment_ids = np.array(segments[local_index], dtype=np.int64, copy=True)
        if loss_mask is None:
            target_mask = np.ones(self.sequence_length, dtype=np.bool_)
        else:
            target_mask = np.array(
                loss_mask[local_index][1:],
                dtype=np.bool_,
                copy=True,
            )

        input_ids = token_ids[:-1]
        labels = token_ids[1:].copy()
        input_segments = segment_ids[:-1]
        same_document = input_segments == segment_ids[1:]
        labels[~(target_mask & same_document)] = -100

        source_name = self.shards[shard_index]['source']
        source_id = self.source_ids[source_name]

        return {
            'input_ids': torch.from_numpy(input_ids),
            'labels': torch.from_numpy(labels),
            'segment_ids': torch.from_numpy(input_segments),
            'source_id': torch.tensor(source_id, dtype=torch.long),
        }


class ShardShuffleSampler(Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_index = 0
        self.num_samples = len(dataset) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def __len__(self):
        return max(0, self.num_samples - self.start_index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start_index(self, start_index):
        self.start_index = min(max(0, start_index), self.num_samples)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            shard_order = torch.randperm(
                len(self.dataset.shards),
                generator=generator,
            ).tolist()
        else:
            shard_order = list(range(len(self.dataset.shards)))

        global_position = 0
        rank_position = 0
        yielded = 0

        for shard_index in shard_order:
            shard_count = self.dataset.shards[shard_index]['num_sequences']
            shard_start = self.dataset.shard_start(shard_index)

            if self.shuffle:
                local_order = torch.randperm(
                    shard_count,
                    generator=generator,
                ).tolist()
            else:
                local_order = range(shard_count)

            for local_index in local_order:
                if global_position >= self.total_size:
                    return

                if global_position % self.num_replicas == self.rank:
                    if rank_position >= self.start_index:
                        yield shard_start + local_index
                        yielded += 1
                        if yielded >= self.num_samples - self.start_index:
                            return
                    rank_position += 1

                global_position += 1
