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
from scipy.stats import entropy
import numpy as np

class AnomalyDetector():
    def __init__(self, model=None, dataset=None, transform=None, model_name=None):
        self.network = model
        self.dataset = dataset
        self.transform = transform
        self.model_name = model_name

    def estimator_worker(self, image):
        image = self.preprocess_image(image)
        output = self.network(image)

        # evaluation for DeepLab only. Panoptic deeplab sends the data already in right format
        if not "anomaly_score" in output[0].keys():
            softmax_out = F.softmax(output[0]['sem_seg'])
            softmax_out = softmax_out.detach().cpu().numpy()
            sem_out = output[0]["sem_seg"].argmax(dim=0).cpu().numpy()
            ent = entropy(softmax_out, axis=0) / np.log(19)
            sem_out[np.where(ent > 0.5)] = 19
            output[0]['anomaly_score'] = torch.tensor(ent)
            output[0]["sem_seg"] = torch.tensor(sem_out)

        '''softmax_out = F.softmax(output[0]['sem_seg'])
        softmax_out = softmax_out.detach().cpu().numpy()
        sem_out = output[0]["sem_seg"].argmax(dim=0).cpu().numpy()
        #sem_out = F.softmax(output[0]['sem_seg'], 0)
        ent = entropy(softmax_out, axis=0) / np.log(19)
        output[0]['anomaly_score'] = torch.tensor(ent)'''
        return output

    def preprocess_image(self, x):

        if self.model_name == "DeepLabV3+_WideResNet38":
            x = torch.unsqueeze(x[0]["image"], dim=0)
        return x

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
    
    start_epoch = 0

    add_panoptic_deeplab_config(cfg)
    
    cfg.merge_from_file("/home/kumarasw/Thesis/meta-ood/src/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml")
    model_name = "Detectron_Panoptic_DeepLab"    
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"

    '''model_name = "Detectron_DeepLab"
    cfg.merge_from_file("/home/kumarasw/Thesis/meta-ood/src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/Detectron_DeepLab_epoch_4_alpha_0.9.pth"'''

    '''model_name = "DeepLabV3+_WideResNet38"
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''
    
    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=model_name, num_classes=19,
                               ckpt_path=init_ckpt, train=False, cfg=cfg)

    # parameter to evaluate OOD using max softmax value
    #network.max_softmax = True

    transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)]) 
    #ds = data_load(root="/export/kiran/cityscapes/", split="val", transform=transform)/home/kumarasw/kiran/cityscapes_coco/cityscapes_val_coco_val
    ds = data_load(root="/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes_ood", split="val", transform=transform)
    detector = AnomalyDetector(network, ds, transform, model_name)
    result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds, collate_fn=panoptic_deep_lab_collate, semantic_only=True)
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
