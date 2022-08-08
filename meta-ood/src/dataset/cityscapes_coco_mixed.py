import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.dataset.coco import COCO
from src.dataset.cityscapes import Cityscapes
from detectron2.projects.panoptic_deeplab.target_generator import PanopticDeepLabTargetGenerator
from detectron2.data import MetadataCatalog
from panopticapi.utils import rgb2id
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from detectron2.data import detection_utils as utils


class CityscapesCocoMix(Dataset):

    def __init__(self, split='train', transform=None,
                 cs_root="/home/datasets/cityscapes",
                 coco_root="/home/datasets/COCO/2017",
                 subsampling_factor=0.1, cs_split=None, coco_split=None, cfg=None, model=None):

        self.transform = transform
        if cs_split is None or coco_split is None:
            self.cs_split = split
            self.coco_split = split
        else:
            self.cs_split = cs_split
            self.coco_split = coco_split

        self.cs = Cityscapes(root=cs_root, split=self.cs_split)
        self.coco = COCO(root=coco_root, split=self.coco_split, proxy_size=int(subsampling_factor*len(self.cs)))
        #self.data_dicts = self.cs.cityscapes_data_dicts + self.coco.coco_data_dicts
        self.data_dicts = self.cs.cityscapes_data_dicts
        self.train_id_out = self.coco.train_id_out
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
        #image = Image.open(data["file_name"]).convert('RGB')
        image = utils.read_image(data["file_name"])
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        target = []
        if self.cfg != None and self.model is not None:
            if data["dataset"] == "cityscapes":
                pan_seg_gt = utils.read_image(data["pan_seg_file_name"], "RGB")
            else:
                pan_seg_gt = utils.read_image(data["pan_seg_file_name"], "L")

            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long")).squeeze().permute(-1, 0, 1)

            if False and self.transform is not None:
                #image, pan_seg_gt = self.transform(image, pan_seg_gt)
                T = Resize(size=(512, 1024), interpolation=InterpolationMode.NEAREST)
                image = T(image)
                pan_seg_gt = T(torch.unsqueeze(pan_seg_gt, dim=0))
                pan_seg_gt = torch.squeeze(pan_seg_gt)
                # Generates training targets for Panoptic-DeepLab.
            if data["dataset"] == "cityscapes":
                targets = self.panoptic_target_generator(rgb2id(pan_seg_gt.permute(1, 2, 0).numpy()), data["segments_info"])
            else:
                targets = self.panoptic_target_generator(pan_seg_gt, data["segments_info"])
            data.update(targets)
            target = targets
            # we don't use default loss calculation from detectron. To avoid error because of OOD value (254)
            data["sem_seg"] = torch.zeros_like(data["sem_seg"])
            data["image"] = image
        elif self.model is not None:
            #sem_seg_gt = Image.open(data["sem_seg_file_name"]).convert('L')
            sem_seg_gt = utils.read_image(data["sem_seg_file_name"], "L").squeeze(2)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.transform is not None:
                #image, sem_seg_gt = self.transform(image, sem_seg_gt)
                T = Resize(size=(512, 1024), interpolation=InterpolationMode.NEAREST)
                image = T(image)
                sem_seg_gt = T(torch.unsqueeze(sem_seg_gt, dim=0))
                sem_seg_gt = torch.squeeze(sem_seg_gt)            
            
            data["sem_seg"] = sem_seg_gt
            target = sem_seg_gt
            # we don't use default loss calculation from detectron. To avoid error because of OOD value (254)
            data["sem_seg"] = torch.zeros_like(data["sem_seg"])

        else:
            target = Image.open(data["sem_seg_file_name"]).convert('L')
            if self.transform is not None:
                data, target = self.transform(image, target)

        data["image"] = image
        data["height"] = image.shape[1]
        data["width"] = image.shape[2]
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

