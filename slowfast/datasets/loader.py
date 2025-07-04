#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
from typing import List
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler

from torch.utils.data import Subset
import random


from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler

from . import utils as utils
from .build import build_dataset


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    time = [item for sublist in time for item in sublist]

    inputs, labels, video_idx, time, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(time),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, time, extra_data
    else:
        return inputs, labels, video_idx, time, extra_data


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    time = default_collate(time)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, time, collated_extra_data

# added by parisa
def construct_continual_loader(cfg):
    split = ['train','val','test']

    # Create base dataset
    dataset = dict()
    dataset['train'] = build_dataset(cfg.TRAIN.DATASET, cfg, split[0])
    dataset['val'] = build_dataset(cfg.TRAIN.DATASET, cfg, split[1])
    dataset['test'] = build_dataset(cfg.TEST.DATASET, cfg, split[2])    

    # Calculate number of classes per task
    num_classes = len(np.unique(dataset['train']._labels))
    classes_per_task = num_classes // cfg.CONTINUAL.NUM_TASKS

    # Generate class indices
    labels = list(range(num_classes))
    if cfg.CONTINUAL.SHUFFLE:
        # added by parisa
        random.seed(42)
        random.shuffle(labels)

    split_datasets = {
        'train': [],
        'val': [],
        'test': []
    }
    class_masks = []

    for task_idx in range(cfg.CONTINUAL.NUM_TASKS):
        if task_idx == cfg.CONTINUAL.NUM_TASKS - 1:
            # Last task gets all remaining classes
            current_classes = labels[task_idx * classes_per_task:]
        else:
            current_classes = labels[task_idx * classes_per_task : (task_idx + 1) * classes_per_task]

        class_masks.append(current_classes)

        # Find indices corresponding to current task's classes
        split_indices = [i for i, target in enumerate(dataset['train']._labels) if target in current_classes]
        split_indices_test = [i for i, target in enumerate(dataset['test']._labels) if target in current_classes]
        split_indices_val = [i for i, target in enumerate(dataset['val']._labels) if target in current_classes]

        # Create subset for current task
        task_dataset_train = Subset(dataset['train'], split_indices)
        task_dataset_test = Subset(dataset['test'], split_indices_test)
        task_dataset_val = Subset(dataset['val'], split_indices_val)

        # Construct loader using your original construct_loader logic
        sampler_train = utils.create_sampler(task_dataset_train, shuffle=True, cfg=cfg)
        sampler_test = utils.create_sampler(task_dataset_test, shuffle=False, cfg=cfg)
        sampler_val = utils.create_sampler(task_dataset_val, shuffle=False, cfg=cfg)
        
        split_datasets['train'].append(torch.utils.data.DataLoader(
            task_dataset_train,
            batch_size=cfg.TRAIN.BATCH_SIZE if split == 'train' else cfg.TEST.BATCH_SIZE,
            shuffle=(False if sampler_train else (split == 'train')),
            sampler=sampler_train,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=(split == 'train'),
            worker_init_fn=utils.loader_worker_init_fn(task_dataset_train),
        ))
        split_datasets['val'].append(torch.utils.data.DataLoader(
            task_dataset_val,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=(False if sampler_val else False),
            sampler=sampler_val,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            worker_init_fn=utils.loader_worker_init_fn(task_dataset_val),
        ))
        split_datasets['test'].append(torch.utils.data.DataLoader(
            task_dataset_test,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=(False if sampler_test else False),
            sampler=sampler_test,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            worker_init_fn=utils.loader_worker_init_fn(task_dataset_test),
        ))


    return split_datasets, class_masks

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test", "test_openset"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test", "test_openset"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        if (
            cfg.MULTIGRID.SHORT_CYCLE
            and split in ["train"]
            and not is_precise_bn
        ):
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            batch_sampler = ShortCycleBatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
            )
            # Create a loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
            )
        else:
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            # Create a loader
            if cfg.DETECTION.ENABLE:
                collate_func = detection_collate
            elif (
                (
                    cfg.AUG.NUM_SAMPLE > 1
                    or cfg.DATA.TRAIN_CROP_NUM_TEMPORAL > 1
                    or cfg.DATA.TRAIN_CROP_NUM_SPATIAL > 1
                )
                and split in ["train"]
                and not cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                collate_func = partial(
                    multiple_samples_collate, fold="imagenet" in dataset_name
                )
            else:
                collate_func = None
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(False if sampler else shuffle),
                sampler=sampler,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=drop_last,
                collate_fn=collate_func,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
            )
    return loader

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.batch_sampler.sampler
            if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
            else loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

    if hasattr(loader.dataset, "prefetcher"):
        sampler = loader.dataset.prefetcher.sampler
        if isinstance(sampler, DistributedSampler):
            # DistributedSampler shuffles data based on epoch
            print("prefetcher sampler")
            sampler.set_epoch(cur_epoch)
