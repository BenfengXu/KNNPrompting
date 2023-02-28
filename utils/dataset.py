import json
import codecs
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import math

import torch
from torch.utils.data import Dataset, Sampler


class BASEDataset:
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__()
        if mode == 'dev':
            mode = 'dev_subsample'
        data_file = os.path.join(data_dir, mode + '.jsonl')
        self.data = []
        with open(data_file, 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                instance = json.loads(line.strip())
                self.data.append(instance)
        # customize your own label map in inheritance
        self.id2label = {0: 'negative', 1: 'positive'}
        self.label2id = {'negative': 0, 'positive': 1}

    def __len__(self):
        return len(self.data)

    def subsamplebyshot(self, n_shot, seed, exclude=None):
        # exclude
        if exclude is not None:
            for ins in exclude:
                self.data.remove(ins)
        # aggregate data by each category
        random.seed(seed)
        data_by_cls = {}
        for i in range(self.__len__()):
            if self.label2id[self.data[i]['label']] not in data_by_cls:
                data_by_cls[self.label2id[self.data[i]['label']]] = []
            data_by_cls[self.label2id[self.data[i]['label']]].append(self.data[i])
        # evenly sample n examples from each category
        data_subsample = []
        for cls in data_by_cls.keys():
            data_subsampled_by_cls = random.sample(data_by_cls[cls], min(n_shot, len(data_by_cls[cls])))
            data_subsample.extend(data_subsampled_by_cls)
        random.shuffle(data_subsample)
        self.data = data_subsample


class SST2Dataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class SUBJDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        # subj only has test set
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'subjective', '1': 'objective'}
        self.id2verb = ['subjective', 'objective']


class AGNEWSDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3}
        self.label2verb = {'1': 'world', '2': 'sports', '3': 'business', '4': 'technology'}
        self.id2verb = ['world', 'sports', 'business', 'technology']


class CBDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        self.label2verb = {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'}
        self.id2verb = ['false', 'true', 'neither']


class CRDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class DBPEDIADataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        if mode == 'dev':
            mode = 'dev_subsample'
        else:
            mode = 'train_subset'  # this is an exception case
        super().__init__(data_dir, mode)
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                         '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                         '11': 10, '12': 11, '13': 12, '14': 13}
        self.label2verb = {'1': 'company', '2': 'school', '3': 'artist', '4': 'athlete', '5': 'politics',
                           '6': 'transportation', '7': 'building', '8': 'nature', '9': 'village', '10': 'animal',
                           '11': 'plant', '12': 'album', '13': 'film', '14': 'book'}
        self.id2verb = ['company', 'school', 'artist', 'athlete', 'politics',
                        'transportation', 'building', 'nature', 'village', 'animal',
                        'plant', 'album', 'film', 'book']


class MPQADataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class MRDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class RTEDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'not_entailment': 0, 'entailment': 1}
        self.label2verb = {'not_entailment': 'false', 'entailment': 'true'}
        self.id2verb = ['false', 'true']


class SST5Dataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        self.label2verb = {'0': 'terrible', '1': 'bad', '2': 'okay', '3': 'good', '4': 'great'}
        self.id2verb = ['terrible', 'bad', 'okay', 'good', 'great']


class TRECDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        self.label2verb = {'0': 'description', '1': 'entity', '2': 'expression', '3': 'human','4': 'location', '5': 'number'}
        self.id2verb = ['description', 'entity', 'expression', 'human', 'location', 'number']