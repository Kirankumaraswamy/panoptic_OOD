import argparse
import time
import os
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
import config as meta_ood_config
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from panoptic_evaluation.evaluation import data_load, data_evaluate
from panoptic_evaluation.cityscapes_ood import CityscapesOOD
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
from scipy.stats import entropy
import numpy as np
from cityscapesscripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt
import ood_config

ckpt_path = ood_config.init_ckpt
threshold = ood_config.threshold
config_file = ood_config.config_file


def get_net():
    """
    Main Function
    """


    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)

    print("Checkpoint file:", ckpt_path)
    network = build_model(cfg)
    DetectionCheckpointer(network).resume_or_load(
        ckpt_path, resume=False
    )

    seg_net = network.cuda()
    seg_net.eval()

    return seg_net


class AnomalyDetector():
    def __init__(self, model=None, num_classes=19):
        self.network = model
        self.num_classes = num_classes

    def estimator_worker(self, image):
        output = self.network(image)

        if ood_config.save_results:
            self.display_results(image, output)

        return output

    def display_results(self, image, output):

        image_save_dir = os.path.join(".", "ood_results")
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        image_save_path = os.path.join(image_save_dir, image[0]["image_id"] + ".png")

        fig = plt.figure(figsize=(20, 14))
        rows = 3
        columns = 3
        images = []
        img1 = np.array(image[0]["real_image"])

        img2 = output[0]["sem_seg"].detach().cpu().squeeze().numpy()

        img3 = output[0]["sem_seg_ood"].detach().cpu().squeeze().numpy()

        img4 = output[0]["anomaly_score"].detach().cpu().squeeze().numpy()

        pan_img = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()

        segment_ids = np.unique(pan_img)
        pan_format = np.zeros(img1.shape, dtype="uint8")
        for segmentId in segment_ids:
            if segmentId > 1000:
                semanticId = segmentId // 1000
                labelInfo = trainId2label[semanticId]
                if labelInfo.hasInstances:
                    mask = np.where(pan_img == segmentId)
                    color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                    pan_format[mask] = color
        img5 = pan_format

        img6 = output[0]["centre_score"].detach().cpu().squeeze().numpy()

        pan_img_ood = output[0]["panoptic_seg_ood"][0].detach().cpu().squeeze().numpy()

        segment_ids = np.unique(pan_img_ood)
        pan_format_ood = np.zeros(img1.shape, dtype="uint8")
        for segmentId in segment_ids:
            if segmentId > 1000:
                semanticId = segmentId // 1000
                labelInfo = trainId2label[semanticId]
                if labelInfo.hasInstances:
                    mask = np.where(pan_img_ood == segmentId)
                    color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                    pan_format_ood[mask] = color
        img7 = pan_format_ood

        ood_ids = [i for i in segment_ids if i >= 19000]
        img8 = np.zeros(pan_img.shape)
        for ood_id in ood_ids:
            ood_mask = np.where(pan_img_ood == ood_id)
            instance_count = ood_id % 19000
            img8[ood_mask] = 1 + instance_count * 10

        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)
        images.append(img6)
        images.append(img7)
        images.append(img8)

        for i in range(8):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        fig.tight_layout()
        plt.savefig(image_save_path)


def panoptic_deep_lab_collate(batch):
    data = [item for item in batch]
    return data


def evaluate(args):
    """Start OoD Evaluation"""
    print("START EVALUATION")

    net = get_net()

    class_mean = np.load(ood_config.class_mean_path, allow_pickle=True)
    class_var = np.load(ood_config.class_var_path, allow_pickle=True)

    transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)])
    ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                   transform=transform)
    net.class_mean = class_mean.item()
    net.class_var = class_var.item()
    net.threshold = ood_config.threshold
    detector = AnomalyDetector(net)
    result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds,
                           collate_fn=panoptic_deep_lab_collate, evaluate_ood=True, semantic_only=False)
    print("====================================================")
    print(result)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        help='possible datasets for statistics; cityscapes')

    args = parser.parse_args()
    print("Command Line Args:", args)

    evaluate(args)
