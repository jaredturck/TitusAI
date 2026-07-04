from dataset import TitusDataset
from data_utils import SequencePacker, ShardWriter, write_manifest


def test_packing_and_boundary_loss_mask(tmp_path):
    output_path = tmp_path / 'train'
    writer = ShardWriter(output_path, 'test', 8, 2)
    packer = SequencePacker(writer, 8)
    packer.add_document([1, 2, 3, 4, 5])
    packer.add_document([6, 7, 8, 9, 10])
    statistics = packer.close()

    write_manifest(
        output_path,
        {'vocab_size': 128, 'name': 'test'},
        8,
        [{'name': 'test', 'train': statistics}],
        statistics['shards'],
    )

    dataset = TitusDataset(output_path)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample['input_ids'].shape == (8,)
    assert sample['labels'].shape == (8,)
    assert sample['segment_ids'].shape == (8,)
    assert sample['labels'][4].item() == -100


def test_instruction_loss_mask(tmp_path):
    output_path = tmp_path / 'train'
    writer = ShardWriter(output_path, 'test', 4, 2)
    packer = SequencePacker(writer, 4)
    packer.add_document(
        [10, 11, 12, 13, 14],
        [0, 0, 1, 1, 1],
    )
    statistics = packer.close()

    write_manifest(
        output_path,
        {'vocab_size': 128, 'name': 'test'},
        4,
        [{'name': 'test', 'train': statistics}],
        statistics['shards'],
    )

    sample = TitusDataset(output_path)[0]
    assert sample['labels'].tolist() == [-100, 12, 13, 14]


def test_shard_sampler_balances_ranks(tmp_path):
    from dataset import ShardShuffleSampler

    output_path = tmp_path / 'train'
    writer = ShardWriter(output_path, 'test', 4, 2)
    packer = SequencePacker(writer, 4)
    for start in range(0, 25, 5):
        packer.add_document([start + value for value in range(5)])
    statistics = packer.close()

    write_manifest(
        output_path,
        {'vocab_size': 128, 'name': 'test'},
        4,
        [{'name': 'test', 'train': statistics}],
        statistics['shards'],
    )

    dataset = TitusDataset(output_path)
    rank_zero = list(ShardShuffleSampler(dataset, 2, 0, True, 5))
    rank_one = list(ShardShuffleSampler(dataset, 2, 1, True, 5))

    assert len(rank_zero) == len(rank_one)
    assert set(rank_zero).isdisjoint(set(rank_one))


def test_shard_sampler_resumes_at_committed_sample(tmp_path):
    from dataset import ShardShuffleSampler

    output_path = tmp_path / 'train'
    writer = ShardWriter(output_path, 'test', 4, 2)
    packer = SequencePacker(writer, 4)
    for start in range(0, 45, 5):
        packer.add_document([start + value for value in range(5)])
    statistics = packer.close()

    write_manifest(
        output_path,
        {'vocab_size': 128, 'name': 'test'},
        4,
        [{'name': 'test', 'train': statistics}],
        statistics['shards'],
    )

    dataset = TitusDataset(output_path)
    full_sampler = ShardShuffleSampler(dataset, 2, 0, True, 19)
    full_order = list(full_sampler)

    resumed_sampler = ShardShuffleSampler(dataset, 2, 0, True, 19)
    resumed_sampler.set_start_index(3)
    resumed_order = list(resumed_sampler)

    assert resumed_order == full_order[3:]


def test_pretraining_shard_without_loss_mask(tmp_path):
    output_path = tmp_path / 'train'
    writer = ShardWriter(
        output_path,
        'test',
        4,
        2,
        store_loss_mask=False,
    )
    packer = SequencePacker(writer, 4)
    packer.add_document([1, 2, 3, 4, 5])
    statistics = packer.close()

    write_manifest(
        output_path,
        {'vocab_size': 128, 'name': 'test'},
        4,
        [{'name': 'test', 'train': statistics}],
        statistics['shards'],
    )

    assert statistics['shards'][0]['loss_mask'] is None
    sample = TitusDataset(output_path)[0]
    assert sample['labels'].tolist() == [2, 3, 4, 5]
