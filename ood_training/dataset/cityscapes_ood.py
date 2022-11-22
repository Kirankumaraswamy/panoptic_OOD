import os
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
import numpy as np
import torch

from detectron2.projects.panoptic_deeplab.target_generator import PanopticDeepLabTargetGenerator
from detectron2.data import MetadataCatalog
from panopticapi.utils import rgb2id
from torchvision.transforms import Resize, RandomCrop
from torchvision.transforms.functional import InterpolationMode
from detectron2.data import detection_utils as utils
import torchvision.transforms.functional as TF
from detectron2.data import transforms as T



class CityscapesOOD(Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
        CityscapesClass('OOD', 50, 19, 'void', 0, True, False, (0, 200, 0)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}


    def __init__(self, root: str = "/home/datasets/cityscapes/", split: str = "val", mode: str = "gtFine",
                 transform: Optional[Callable] = None, cfg=None) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        if transform is not None:
            self.augmentations = T.AugmentationList(transform)
        else:
            self.augmentations = None
        self.images = []
        self.targets = []
        self.predictions = []
        if self.split == "train":
            self.cityscapes_data_dicts = DatasetCatalog.get("cityscapes_fine_panoptic_train")
        elif self.split == "test":
            self.cityscapes_data_dicts = DatasetCatalog.get("cityscapes_fine_panoptic_test")
        else:
            self.cityscapes_data_dicts = DatasetCatalog.get("cityscapes_fine_panoptic_val")

        # needed for panoptic training
        dataset_names = 'cityscapes_fine_panoptic_train'
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())

        self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=thing_ids,
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

    def __getitem__(self, i):
        data = self.cityscapes_data_dicts[i]
        # image = Image.open(data["file_name"]).convert('RGB')
        image = utils.read_image(data["file_name"])

        target = []
        pan_seg_gt = utils.read_image(data["pan_seg_file_name"], "RGB")

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
        if self.augmentations is not None:
            _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg


        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        #pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long")).squeeze().permute(-1, 0, 1)

        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt),
                                                 data["segments_info"])
        data.update(targets)
        target = targets
        '''import matplotlib.pyplot as plt
        plt.imshow(torch.permute(image, (1, 2, 0)).numpy())
        plt.show()
        plt.imshow(target["sem_seg"].numpy())
        plt.show()
        plt.imshow(target["center"].numpy())
        plt.show()'''
        data["image"] = image
        data["height"] = 1024
        data["width"] = 2048
        return data, target

    def __len__(self) -> int:
        return len(self.cityscapes_data_dicts)
