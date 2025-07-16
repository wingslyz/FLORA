import sys, os
import pickle


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def prepare_data_PACS(cfg, data_base_path):
    data_base_path = data_base_path
    transform_office = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # amazon
    art_painting_trainset = PACSDataset(data_base_path, 'art_painting', transform=transform_office)
    lab2cname = art_painting_trainset.lab2cname
    classnames = art_painting_trainset.classnames
    art_painting_trainset = art_painting_trainset.data_detailed
    art_painting_testset = PACSDataset(data_base_path, 'art_painting', transform=transform_test, train=False).data_detailed
    # caltech
    cartoon_trainset = PACSDataset(data_base_path, 'cartoon', transform=transform_office).data_detailed
    cartoon_testset = PACSDataset(data_base_path, 'cartoon', transform=transform_test, train=False).data_detailed
    # dslr
    photo_trainset = PACSDataset(data_base_path, 'photo', transform=transform_office).data_detailed
    photo_testset = PACSDataset(data_base_path, 'photo', transform=transform_test, train=False).data_detailed
    # webcam
    sketch_trainset = PACSDataset(data_base_path, 'sketch', transform=transform_office).data_detailed
    sketch_testset = PACSDataset(data_base_path, 'sketch', transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(art_painting_trainset), len(cartoon_trainset), len(photo_trainset), len(sketch_trainset)]
    test_data_num_list = [len(art_painting_testset), len(cartoon_testset), len(photo_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [sketch_trainset, art_painting_trainset, cartoon_trainset,photo_trainset]
    # test_set = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]
    # test_set = [amazon_trainset + amazon_testset, caltech_trainset + caltech_testset, dslr_trainset + dslr_testset,webcam_trainset + webcam_testset]
    test_set = [sketch_testset,art_painting_testset,cartoon_testset,photo_testset ]
    return train_set, test_set, classnames, lab2cname


def prepare_data_PACS_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    K = 7
    min_require_size = 2
    n_clients = 3
    # amazon
    print("art_painting: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_PACS(data_base_path, 'art_painting', cfg.DATASET.BETA,
                                                                           n_parties=n_clients,
                                                                           min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office(data_base_path,'amazon', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=min_img_num)
    # net_dataidx_map_test = Adjust_test_dataset_office('amazon', train_ratio[0])
    art_painting_trainset = [[] for i in range(n_clients)]
    art_painting_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        art_painting_trainset[i] = PACSDataset_sub(data_base_path, 'art_painting', net_dataidx_map_train[i],
                                               transform=transform_train)
        art_painting_testset[i] = PACSDataset_sub(data_base_path, 'art_painting', net_dataidx_map_test[i], train=False,
                                              transform=transform_test).data_detailed
        lab2cname = art_painting_trainset[i].lab2cname
        classnames = art_painting_trainset[i].classnames
        art_painting_trainset[i] = art_painting_trainset[i].data_detailed

    # caltech
    print("cartoon: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_PACS(data_base_path, 'cartoon', cfg.DATASET.BETA,
                                                                           n_parties=n_clients,
                                                                           min_require_size=min_require_size)
    # caltech_trainset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train, transform=transform_train).data_detailed
    # caltech_testset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test, transform=transform_train).data_detailed
    cartoon_trainset = [[] for i in range(n_clients)]
    cartoon_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        cartoon_trainset[i] = PACSDataset_sub(data_base_path, 'cartoon', net_dataidx_map_train[i],
                                                transform=transform_train).data_detailed
        cartoon_testset[i] = PACSDataset_sub(data_base_path, 'cartoon', net_dataidx_map_test[i], train=False,
                                               transform=transform_test).data_detailed

    # dslr
    print("photo: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_PACS(data_base_path, 'photo', cfg.DATASET.BETA,
                                                                           n_parties=n_clients,
                                                                           min_require_size=min_require_size)
    # dslr_trainset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train, transform=transform_train).data_detailed
    # dslr_testset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_test, transform=transform_train).data_detailed
    photo_trainset = [[] for i in range(n_clients)]
    photo_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        photo_trainset[i] = PACSDataset_sub(data_base_path, 'photo', net_dataidx_map_train[i],
                                             transform=transform_train).data_detailed
        photo_testset[i] = PACSDataset_sub(data_base_path, 'photo', net_dataidx_map_test[i], train=False,
                                            transform=transform_test).data_detailed

    # webcam
    print("sketch: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_PACS(data_base_path, 'sketch', cfg.DATASET.BETA,
                                                                           n_parties=n_clients,
                                                                           min_require_size=min_require_size)

    sketch_trainset = [[] for i in range(n_clients)]
    sketch_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        sketch_trainset[i] = PACSDataset_sub(data_base_path, 'sketch', net_dataidx_map_train[i],
                                               transform=transform_train).data_detailed
        sketch_testset[i] = PACSDataset_sub(data_base_path, 'sketch', net_dataidx_map_test[i], train=False,
                                              transform=transform_test).data_detailed

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in [art_painting_trainset, cartoon_trainset, photo_trainset, sketch_trainset]:
        for i in range(n_clients):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in [art_painting_testset, cartoon_testset, photo_testset, sketch_testset]:
        for i in range(n_clients):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)


class PACSDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'PACS/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'PACS/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)

        self.site_domian = {'art_painting': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}
        self.domain = self.site_domian[site]
        self.lab2cname = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4,
                          'house': 5, 'person': 6}
        self.classnames = {'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'}

        self.target = [self.lab2cname[text] for text in self.label]
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.target))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.target))
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return Datum(impath=image, label=int(label), domain=self.domain, classname=self.label[idx])

def Dataset_partition_PACS(base_path, site, beta, split_test=True, n_parties=3, min_require_size=2):
    min_size = 0
    K = 7
    # np.random.seed(2023)

    train_path = os.path.join(base_path,'PACS/{}_train.pkl'.format(site))
    test_path = os.path.join(base_path,'PACS/{}_test.pkl'.format(site))
    _, train_text_labels = np.load(train_path, allow_pickle=True)
    _, test_text_labels = np.load(test_path, allow_pickle=True)

    label_dict={'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4,
                          'house': 5, 'person': 6}
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    N_train = train_labels.shape[0]
    N_test = test_labels.shape[0]
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            # proportions = proportions * 2
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            test_part_list = np.split(test_idx_k, proportions_test)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test,min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(site, "Training data split: ",traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(site, "Testing data split: ",testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test

def Adjust_test_dataset_PACS(site, class_ratio):
    c_num = 7
    label_dict={'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4,
                          'house': 5, 'person': 6}
    _, test_text_labels = np.load('autodl-tmp/FedPGP/DATA/PACS/{}_test.pkl'.format(site), allow_pickle=True)
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


class PACSDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'PACS/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'PACS/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)

        self.site_domian = {'art_painting': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}
        self.domain = self.site_domian[site]
        self.lab2cname = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4,
                          'house': 5, 'person': 6}
        self.classnames = {'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'}
        self.target = np.asarray([self.lab2cname[text] for text in self.label])
        self.target = self.target[net_dataidx_map]
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def second_divide(self, partitions):
        self.target = self.target[partitions]

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts