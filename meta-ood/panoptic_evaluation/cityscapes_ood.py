import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from panopticapi.utils import rgb2id
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from detectron2.data import detection_utils as utils
import os
import json


class CityscapesOOD(Dataset):

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, cs_root=None, split='val', transform=None):

        self.transform = transform
        self.split = split
        self.root = cs_root
        

        self.json_file = os.path.join(self.root, "gtFine", "cityscapes_panoptic_"+self.split+".json")
        with open(self.json_file, 'r') as f:
            self.cityscapes_data_dict = json.load(f)["annotations"]
        
        for dict in self.cityscapes_data_dict:
            image_id = dict["image_id"]
            city_name = image_id.split("_")[0]
            dict["sem_seg_file_name"] = os.path.join(self.root, "gtFine", self.split, city_name, image_id +"_gtFine_labelTrainIds.png")
            dict["pan_seg_file_name"] = os.path.join(self.root, "gtFine", "cityscapes_panoptic_"+self.split, image_id +"_gtFine_panoptic.png")
            dict["file_name"] = os.path.join(self.root, "leftImg8bit", self.split, city_name, image_id+"_leftImg8bit.png")


    def __getitem__(self, i):
        data = self.cityscapes_data_dict[i]
        image = Image.open(data["file_name"]).convert('RGB')
        #image = utils.read_image(data["file_name"])
        #image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.transform is not None:
            image = self.transform(image)  
        data["image"] = image
        data["height"] = image.shape[1]
        data["width"] = image.shape[2]
        return data

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.cityscapes_data_dict)



