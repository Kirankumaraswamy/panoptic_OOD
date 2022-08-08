import argparse
import time
import os
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
import config as meta_ood_config
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from src.model_utils import load_network
from detectron2.engine import DefaultTrainer, default_argument_parser
from panoptic_evaluation.evaluation import data_load, data_evaluate
from panoptic_evaluation.cityscapes_ood import CityscapesOOD

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize

class AnomalyDetector():
    def __init__(self, model=None, dataset=None, transform=None):
        self.network = model
        self.dataset = dataset
        self.transform = transform

    def estimator_worker(self, image):
        output = self.network(image)
        return output

def panoptic_deep_lab_collate(batch):
    data = [item for item in batch]
    return data

def evaluate(args):
    """Start OoD Evaluation"""
    print("START EVALUATION")
    start = time.time()

    # This parameter is needed for panoptic segmentation data loading
    dataset_cfg = None

    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file("/home/kumarasw/Thesis/meta-ood/src/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml")
    
    start_epoch = 0
    model_name = "Detectron_Panoptic_DeepLab"
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=model_name, num_classes=19,
                               ckpt_path=init_ckpt, train=False, cfg=cfg)

    
    transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)]) 
    #ds = data_load(root="/export/kiran/cityscapes/", split="val", transform=transform)/home/kumarasw/kiran/cityscapes_coco/cityscapes_val_coco_val
    ds = data_load(root="/home/kumarasw/kiran/cityscapes_coco/cityscapes_ood", split="val", transform=transform)
    detector = AnomalyDetector(network, ds, transform)
    result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds, collate_fn=panoptic_deep_lab_collate)
    print("====================================================")
    print(result)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)

    # use detectron2 distributed args
    args = default_argument_parser().parse_args()
    # args from current file
    args.default_args = vars(parser.parse_args())
    print("Command Line Args:", args)

    evaluate(args)
