import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.dataset.coco import COCO
from src.dataset.cityscapes import Cityscapes
from detectron2.projects.panoptic_deeplab.target_generator import PanopticDeepLabTargetGenerator
from detectron2.data import MetadataCatalog
from panopticapi.utils import rgb2id

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random
import io
import cv2
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode



class CityscapesCocoMix(Dataset):

    def __init__(self, split='train', transform=None,
                 cs_root="/home/datasets/cityscapes",
                 coco_root="/home/datasets/COCO/2017",
                 subsampling_factor=0.1, cs_split=None, coco_split=None, cfg=None, model=None):

        self.transform = transform
        if cs_split is None:
            self.cs_split = split
        else:
            self.cs_split = cs_split

        self.cs = Cityscapes(root=cs_root, split=self.cs_split)
        self.data_dicts = self.cs.cityscapes_data_dicts
        self.num_classes = self.cs.num_train_ids
        self.mean = self.cs.mean
        self.std = self.cs.std
        self.void_ind = self.cs.ignore_in_eval_ids
        self.cfg = cfg
        self.model = model


        if self.cfg != None:
            # needed for panoptic training
            dataset_names = 'cityscapes_fine_panoptic_train'
            dataset_names = cfg.DATASETS.TRAIN
            meta = MetadataCatalog.get(dataset_names[0])
            thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())
            # 254 is out of distribution object in coco dataset
            thing_ids.append(254)
            self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
                ignore_label=meta.ignore_label,
                thing_ids= thing_ids,
                sigma=cfg.INPUT.GAUSSIAN_SIGMA,
                ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
                small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
                small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
                ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
            )

    def __getitem__(self, i):
        data = self.data_dicts[i]
        image = Image.open(data["file_name"]).convert('RGB')
        target = []
        if self.cfg != None and self.model is not None:
            if data["dataset"] == "cityscapes":
                pan_seg_gt= Image.open(data["pan_seg_file_name"]).convert("RGB")
            else:
                pan_seg_gt = Image.open(data["pan_seg_file_name"]).convert("L")
            if self.transform is not None:
                image, pan_seg_gt = self.transform(image, pan_seg_gt)
            # Generates training targets for Panoptic-DeepLab.
            if data["dataset"] == "cityscapes":
                targets = self.panoptic_target_generator(rgb2id(pan_seg_gt.numpy()), data["segments_info"])
            else:
                targets = self.panoptic_target_generator(pan_seg_gt, data["segments_info"])
            data.update(targets)
            target = targets["sem_seg"]
            # we don't use default loss calculation from detectron. To avoid error because of OOD value (254)
            data["sem_seg"] = torch.zeros_like(data["sem_seg"])
            data["image"] = image
        elif self.model is not None:
            sem_seg_gt = Image.open(data["sem_seg_file_name"]).convert('L')
            if self.transform is not None:
                image, sem_seg_gt = self.transform(image, sem_seg_gt)

            # add random mask into the image
            image = self.add_random_mask(image)


            data["sem_seg"] = sem_seg_gt
            target = sem_seg_gt
            # we don't use default loss calculation from detectron. To avoid error because of OOD value (254)
            #data["sem_seg"] = torch.zeros_like(data["sem_seg"])
            data["image"] = image
        else:
            target = Image.open(data["sem_seg_file_name"]).convert('L')
            if self.transform is not None:
                data, target = self.transform(image, target)
        return data, target

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.data_dicts)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.cs)
        fmt_str += 'COCO Split: %s\n' % self.coco_split
        fmt_str += '----Number of images: %d\n' % len(self.coco)
        return fmt_str.strip()

    def add_random_mask(self, image):
        height = image.size()[-2]
        width = image.size()[-1]
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

        random_width = random.randint(30, int(height/4))
        random_height = random.randint(30, int(width/6))

        im_out = torch.unsqueeze(torch.tensor(im_out), dim=0)
        T = Resize(size=(random_height, random_width), interpolation=InterpolationMode.NEAREST)
        im_out = T(im_out)
        im_out = im_out.squeeze().numpy()

        h, w = im_out.shape
        mask = np.zeros((3, height, width), dtype="float64")

        end_width = width - w
        end_height = height - h
        start_width = random.randint(0, end_width)
        start_height = random.randint(0, end_height)

        mask[:, start_height:start_height + h, start_width:start_width + w] += im_out
        mask[np.where(mask == 0.0)] = 1.0
        mask[np.where(mask == 255.0)] = 0.0

        image = image * torch.tensor(mask)

        # add random pixel values to masked image Normalize(dataset.mean, dataset.std)
        in_pixels = np.where(mask == 1.0)
        ood_pixels = np.where(mask == 0.0)
        ood_size = ood_pixels[0].size
        random_pixels = np.random.rand(ood_size)
        mask[ood_pixels] = random_pixels
        mask[in_pixels] = 0.0
        image = image + torch.tensor(mask)

        return image