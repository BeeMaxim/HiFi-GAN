from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["audio"] = pad_sequence([x["audio"].transpose(0, 1) for x in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["audio_path"] = [x["audio_path"] for x in dataset_items]
    result_batch["text"] = [x["text"] for x in dataset_items]
    result_batch["audio_len"] = [x["audio_len"] for x in dataset_items]

    return result_batch
