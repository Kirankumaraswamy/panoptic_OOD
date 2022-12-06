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
        rows = 2
        columns = 3
        images = []
        img1 = np.array(image[0]["real_image"])

        img2 = output[0]["sem_seg"].detach().cpu().squeeze().numpy()

        img3 = output[0]["anomaly_score"].detach().cpu().squeeze().numpy()

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
        img4 = pan_format

        img5 = output[0]["centre_score"].detach().cpu().squeeze().numpy()

        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)

        for i in range(5):
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

    net.sum_all_class_mean = np.load(ood_config.sum_all_class_mean_path, allow_pickle=True).item()
    net.sum_all_class_var = np.load(ood_config.sum_all_class_var_path, allow_pickle=True).item()
    net.correct_class_mean = np.load(ood_config.correct_class_mean_path, allow_pickle=True).item()
    net.correct_class_var = np.load(ood_config.correct_class_var_path, allow_pickle=True).item()
    net.evaluate_ood = True

    transform = None
    thresholds = [0.02* i for i in range(0,4)]
    #threshold = ood_config.threshold
    specificity = []
    sensitivity = []
    gmean = []

    for threshold in thresholds:
        print("====================================================")
        print("              Threshold: ", threshold)
        print("====================================================")

        ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                       transform=transform)
        net.ood_threshold = threshold
        detector = AnomalyDetector(net)
        result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds,
                               collate_fn=panoptic_deep_lab_collate, batch_size=1, evaluate_ood=True, semantic_only=False)

        print(result)
        specificity.append(result['semantic_seg']['sem_seg']['uSpecificity'])
        sensitivity.append(result['semantic_seg']['sem_seg']['uSensitivity'])
        gmean.append(result['semantic_seg']['sem_seg']['uGmean'])

    if len(thresholds) > 1:
        fig = plt.figure()
        plt.plot(thresholds, specificity,  label="uSpecificity")
        plt.plot(thresholds, sensitivity,  label="uSensitivity")
        plt.plot(thresholds, gmean, label="uGmean")
        plt.legend()
        fig.savefig("./sensitivity_vs_specificity.png")

    print("Thresholds: ", thresholds)
    print("Gmean: ", gmean)
    print('Usensitivity: ', sensitivity)
    print("Uspecivicity: ", specificity)



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
