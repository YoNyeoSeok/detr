from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pdb
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import json
import cv2
from PIL import Image
import util
torch.multiprocessing.set_sharing_strategy('file_system')


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, img_folder, train_file, class_list, verb_path, role_path, verb_info, inference=False, inference_verbs=None, transform=None, is_visualizing=False):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.img_folder = img_folder
        self.inference = inference
        self.inference_verbs = inference_verbs
        self.train_file = train_file
        self.verb_path = verb_path
        self.role_path = role_path
        self.class_path = class_list
        self.transform = transform
        self.is_visualizing = is_visualizing

        with open(self.verb_path, 'r') as f:
            self.verb_to_idx, self.idx_to_verb = self.load_verb(f)
            self.num_verbs = len(self.verb_to_idx)
        with open(self.role_path, 'r') as f:
            self.role_to_idx, self.idx_to_role = self.load_role(f)
            self.num_roles = len(self.role_to_idx)
        with open(self.class_path, 'r') as file:
            self.noun_to_idx, self.idx_to_noun = self.load_classes(csv.reader(file, delimiter=','))
            self.num_nouns = len(self.noun_to_idx) - 1  # padding noun last
            self.pad_noun = len(self.noun_to_idx) - 1

        # verb_role
        self.verb_role = {verb: value['order'] for verb, value in verb_info.items()}
        self.vidx_ridx = [[self.role_to_idx[role] for role in self.verb_role[verb]] for verb in self.idx_to_verb]

        if not self.inference:
            with open(self.train_file) as file:
                train_json = json.load(file)
            self.image_data = self._read_annotations(train_json, self.noun_to_idx)
            self.image_names = list(self.image_data.keys())
        else:
            self.image_names = []
            with open(train_file) as f:
                for line in f:
                    self.image_names.append(line.split('\n')[0])

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

        # verb role adjacency matrix
        self.verb_role_adj_matrix = np.tile(np.identity(len(self.role_to_idx)),
                                            (len(self.verb_to_idx), 1, 1)).astype(bool)
        for vidx, ridx in enumerate(self.vidx_ridx):
            ridx = np.array(ridx)
            self.verb_role_adj_matrix[vidx:vidx + 1, ridx[:, None], ridx] = np.ones(len(ridx)).astype(bool)

        # role adjacency matrix
        self.role_adj_matrix = self.verb_role_adj_matrix.any(0)

    def load_classes(self, csv_reader):
        result = {}
        idx_to_result = []
        for line, row in enumerate(csv_reader):
            line += 1
            class_name, class_id = row
            class_id = int(class_id)
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            idx_to_result.append(class_name.split('_')[0])

        return result, idx_to_result

    def load_verb(self, file):
        verb_to_idx = {}
        idx_to_verb = []

        k = 0
        for line in file:
            verb = line.split('\n')[0]
            idx_to_verb.append(verb)
            verb_to_idx[verb] = k
            k += 1
        return verb_to_idx, idx_to_verb

    def load_role(self, file):
        role_to_idx = {}
        idx_to_role = []

        k = 0
        for line in file:
            role = line.split('\n')[0]
            idx_to_role.append(role)
            role_to_idx[role] = k
            k += 1
        return role_to_idx, idx_to_role

    def make_dummy_annot(self):
        return np.zeros((len(self.role_to_idx), 3))

    def __len__(self):
        # return 16
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        if self.inference:
            verb_idx = self.inference_verbs[self.image_names[idx]]
            annot = self.make_dummy_annot()
            sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
            if self.transform:
                sample = self.transform(sample)
            return sample

        annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[2]
        verb = verb.split('_')[0]

        verb_idx = self.verb_to_idx[verb]
        verb_role_idx = self.vidx_ridx[verb_idx]
        sample = {'img': img, 'annot': annot,
                  'img_name': self.image_names[idx], 'verb_idx': verb_idx, 'verb_role_idx': verb_role_idx}
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        return sample

    def load_image(self, image_index):

        im = Image.open(self.image_names[image_index])
        im = im.convert('RGB')
        return im

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 3))

        # parse annotations
        for a in annotation_list:
            annotation = np.zeros((1, 3))  # allow for 3 annotations

            annotation[0, 0] = self.noun_to_idx[a['class1']]
            annotation[0, 1] = self.noun_to_idx[a['class2']]
            annotation[0, 2] = self.noun_to_idx[a['class3']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, json, classes):
        dummy_annot = [{f"class{n}": 'Pad' for n in [1, 2, 3]} for role in self.role_to_idx.keys()]

        result = {}
        for image in json:
            verb = json[image]['verb']
            frames = json[image]['frames']
            img_file = f"{self.img_folder}/" + image

            result[img_file] = dummy_annot.copy()
            for role in self.verb_role[verb]:
                class1 = frames[0][role]
                class2 = frames[1][role]
                class3 = frames[2][role]
                if class1 == '':
                    class1 = 'blank'
                if class2 == '':
                    class2 = 'blank'
                if class3 == '':
                    class3 = 'blank'
                if class1 not in classes:
                    class1 = 'oov'
                if class2 not in classes:
                    class2 = 'oov'
                if class3 not in classes:
                    class3 = 'oov'

                ridx = self.role_to_idx[role]
                result[img_file][ridx] = {
                    'class1': class1,
                    'class2': class2,
                    'class3': class3
                }
        return result

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)
    verb_role_indices = [s['verb_role_idx'] for s in data]
    verb_role_indices = [torch.tensor(vri) for vri in verb_role_indices]

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 3)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.tensor(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 3)) * -1

    return (util.misc.nested_tensor_from_tensor_list(imgs),
            [{'verb': vi,
              'roles': vri,
              'labels': annot,
              'image_name': img_name}
             for vi, vri, annot, img_name in zip(verb_indices, verb_role_indices, annot_padded, img_names)])


def build(image_set, args):
    root = Path(args.imsitu_path)
    img_folder = root / args.image_dir

    PATHS = {
        "train": root / "train.json",
        "val": root / "dev.json",
        "test": root / "test.json",
    }
    ann_file = PATHS[image_set]

    classes_file = Path(args.imsitu_path) / "train_classes.csv"
    verb_path = Path(args.imsitu_path) / "verb_indices.txt"
    role_path = Path(args.imsitu_path) / "role_indices.txt"

    with open(f'{args.imsitu_path}/imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    TRANSFORMS = {
        "train": transforms.Compose([
            transforms.Resize(512),  # min edge 512 resize
            transforms.RandomRotation(0)
            ] + ([] if args.remove_crop else [
                transforms.RandomCrop(512),  # square crop
            ]) + [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer,
        ]),
        "val": transforms.Compose([
            transforms.Resize(512),
            ] + ([] if args.remove_crop else [
                transforms.CenterCrop(512),  # center square crop
            ]) + [
            transforms.ToTensor(),
            normalizer,
        ]),
        "test": transforms.Compose([
            transforms.Resize(512),
            ] + ([] if args.remove_crop else [
                transforms.CenterCrop(512),  # center square crop
            ]) + [
            transforms.ToTensor(),
            normalizer,
        ]),
    }
    tfs = TRANSFORMS[image_set]

    dataset = CSVDataset(img_folder=str(img_folder),
                         train_file=ann_file,
                         class_list=classes_file,
                         verb_path=verb_path,
                         role_path=role_path,
                         verb_info=verb_orders,
                         transform=tfs)

    args.vr_adj_mat = dataset.verb_role_adj_matrix
    args.num_verbs = dataset.num_verbs
    args.num_roles = dataset.num_roles
    args.num_nouns = dataset.num_nouns
    args.pad_noun = dataset.pad_noun

    return dataset
