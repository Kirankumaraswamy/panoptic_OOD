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
from panopticapi.utils import rgb2id, id2rgb
from torchvision.transforms import Resize, RandomCrop
from torchvision.transforms.functional import InterpolationMode
from detectron2.data import detection_utils as utils
import torchvision.transforms.functional as TF
from detectron2.data import transforms as T

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random
import io
import cv2
from torchvision.transforms import Resize
import json
import copy




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
        CityscapesClass('OOD', 50, 19, 'void', 0, True, False, (255, 255, 255)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    label_to_id = {}
    label_to_train_id = {}
    id_to_train_id = {}

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        label_to_id[labels[i].name] = labels[i].id
        label_to_train_id[labels[i].name] = labels[i].train_id
        id_to_train_id[labels[i].id] = labels[i].train_id

        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}


    def __init__(self, root: str = "/home/datasets/cityscapes/", split: str = "val", mode: str = "gtFine",
                 transform: Optional[Callable] = None, cfg=None, dataset="cityscapes", ood_training=False) -> None:
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

        self.cityscapes_data_dicts = []

        if dataset == "bdd":
            self.dataset_prefix = "bdd"
        else:
            self.dataset_prefix = "cityscapes"

        self.json_file = os.path.join(self.root, "gtFine", self.dataset_prefix+"_panoptic_" + self.split + ".json")
        with open(self.json_file, 'r') as f:
            self.cityscapes_data_dicts = json.load(f)["annotations"]

        for dict in self.cityscapes_data_dicts:
            image_id = dict["image_id"]

            city_name = image_id.split("_")[0]
            dict["sem_seg_file_name"] = os.path.join(self.root, "gtFine", self.split, city_name,
                                                     image_id + "_gtFine_labelTrainIds.png")
            dict["pan_seg_file_name"] = os.path.join(self.root, "gtFine", self.dataset_prefix+"_panoptic_" + self.split,
                                                     image_id + "_gtFine_panoptic.png")
            dict["file_name"] = os.path.join(self.root, "leftImg8bit", self.split, city_name,
                                             image_id + "_leftImg8bit.png")
            for segments in dict["segments_info"]:
                segments['category_id'] = self.id_to_train_id[segments['category_id']]

            # needed for panoptic training
        dataset_names = 'cityscapes_fine_panoptic_train'
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())
        self.ood_training = ood_training

        self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=thing_ids,
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
            ood_training=ood_training
        )

    def __getitem__(self, i):
        data = copy.deepcopy(self.cityscapes_data_dicts[i])
        # image = Image.open(data["file_name"]).convert('RGB')
        image = utils.read_image(data["file_name"])

        target = []
        pan_seg_gt = utils.read_image(data["pan_seg_file_name"], "RGB")

        #image, pan_seg_gt, data = self.add_random_mask(image, pan_seg_gt.copy(), data.copy())

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
        if self.augmentations is not None:
            _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg


        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        #pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long")).squeeze().permute(-1, 0, 1)

        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), data["segments_info"])
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
        c,h,w =image.size()
        data["height"] = h
        data["width"] = w
        return data, target

    def __len__(self) -> int:
        return len(self.cityscapes_data_dicts)

    def add_random_mask(self, image,  pan_seg_gt, data_dict):
        height, width, c = image.shape

        random_instances = random.randint(2, 6)

        for i in range(random_instances):
            instance_id = 1000 * self.label_to_id["OOD"] + i
            n = random.randint(2, 9)  # Number of possibly sharp edges
            r = random.random()  # magnitude of the perturbation from the unit circle,
            # should be between 0 and 1
            N = n * 3 + 1  # number of points in the Path
            # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

            angles = np.linspace(0, 2 * np.pi, N)
            codes = np.full(N, Path.CURVE4)
            codes[0] = Path.MOVETO
            verts = np.stack((np.cos(angles), np.sin(angles))).T * (2 * r * np.random.random(N) + 1 - r)[:, None]

            verts[-1, :] = verts[0, :]  # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
            path = Path(verts, codes)

            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

            ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
            ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
            ax.axis('off')  # removes the axis to leave only the shape

            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')

                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            im = data.reshape((int(h), int(w), -1))[:, :, 0]
            plt.figure(1).clear()

            th, im_th = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY_INV)

            # Copy the thresholded image.
            im_floodfill = im_th.copy()
            h, w = im_th.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)

            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0, 0), 255)
            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            # Combine the two images to get the foreground.
            im_out = im_th | im_floodfill_inv

            random_width = random.randint(30, int(height/2))
            random_height = random.randint(30, int(width/3))

            im_out = torch.unsqueeze(torch.tensor(im_out), dim=0)
            T = Resize(size=(random_height, random_width), interpolation=InterpolationMode.NEAREST)
            im_out = T(im_out)
            im_out = im_out.squeeze().numpy()

            h, w = im_out.shape
            mask = np.zeros((3, height, width), dtype="uint8")

            end_width = width - w
            end_height = height - h
            start_width = random.randint(0, end_width)
            start_height = random.randint(0, end_height)

            mask[:, start_height:start_height + h, start_width:start_width + w] += im_out
            mask[np.where(mask == 0)] = 1
            mask[np.where(mask == 255)] = 0

            mask = mask.transpose((1, 2, 0))

            image = image * mask

            # add random pixel values to masked image Normalize(dataset.mean, dataset.std)
            in_pixels = np.where(mask[:,:,0] == 1)
            ood_pixels = np.where(mask[:,:,0] == 0)
            rgb_random = np.random.randint(0, 255, 3)
            r_random_pixels = np.random.randint(max(0, rgb_random[0] - 20), min(255, rgb_random[0] + 20), ood_pixels[0].size)
            g_random_pixels = np.random.randint(max(0, rgb_random[1] - 20), min(255, rgb_random[1] + 20), ood_pixels[0].size)
            b_random_pixels = np.random.randint(max(0, rgb_random[2] - 20), min(255, rgb_random[2] + 20),ood_pixels[0].size)

            mask[ood_pixels[0], ood_pixels[1], 0] = r_random_pixels
            mask[ood_pixels[0], ood_pixels[1], 1] = g_random_pixels
            mask[ood_pixels[0], ood_pixels[1], 2] = b_random_pixels
            mask[in_pixels[0], in_pixels[1], :] = 0
            image = image + mask

            panoptic_rgb_mask = id2rgb(instance_id)
            pan_seg_gt[ood_pixels[0], ood_pixels[1], 0] = panoptic_rgb_mask[0]
            pan_seg_gt[ood_pixels[0], ood_pixels[1], 1] = panoptic_rgb_mask[1]
            pan_seg_gt[ood_pixels[0], ood_pixels[1], 2] = panoptic_rgb_mask[2]

            segment_info = {
                "area": ood_pixels[0].size,
                "category_id": self.label_to_train_id["OOD"],
                "id": instance_id,
                "iscrowd": 0,
                "is_ood": True,
            }

            data_dict['segments_info'].append(segment_info)

        return image, pan_seg_gt, data_dict
