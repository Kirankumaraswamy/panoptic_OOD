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
from cityscapesscripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt
import ood_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

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
            sem_out_ood = sem_out.copy()
            ent = entropy(softmax_out, axis=0) / np.log(19)
            sem_out_ood[np.where(ent > 0.5)] = 19
            output[0]['anomaly_score'] = torch.tensor(ent)
            output[0]["sem_seg_ood"] = torch.tensor(sem_out_ood)
            output[0]["sem_seg"] = torch.tensor(sem_out)

        plt.imshow(torch.squeeze(output[0]["anomaly_score"].detach().cpu()).numpy())
        plt.show()
       
        if ood_config.save_results:
            self.display_results(image, output)
        return output

    def preprocess_image(self, x):

        if self.model_name == "DeepLabV3+_WideResNet38":
            x = torch.unsqueeze(x[0]["image"], dim=0)
        return x

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
    start = time.time()

    # This parameter is needed for panoptic segmentation data loading
    dataset_cfg = None

    # load configuration from cfg files for detectron2
    cfg = get_cfg()

    model_name = ood_config.model_name
    ckpt_path = ood_config.init_ckpt
    max_softmax = ood_config.max_softmax

    
    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)

    if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
        cfg = get_cfg()
        if model_name == "Detectron_DeepLab":
            add_deeplab_config(cfg)
            cfg.merge_from_file(ood_config.config_file)
        elif model_name == "Detectron_Panoptic_DeepLab":
            add_panoptic_deeplab_config(cfg)
            cfg.merge_from_file(ood_config.config_file)
        network = build_model(cfg)
        #network = torch.nn.DataParallel(network).cuda()
        DetectionCheckpointer(network).resume_or_load(
            ckpt_path, resume=False
        )
    else:
        network = nn.DataParallel(DeepWV3Plus(num_classes))
        network.load_state_dict(torch.load(ckpt_path)['model'], strict=False)

    network = network.cuda()
    network.eval()


    # parameter to evaluate OOD using max softmax value
    if ood_config.evaluate_ood:
        network.max_softmax = max_softmax
        network.evaluate_ood = ood_config.evaluate_ood

    transform = None
    thresholds = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    #thresholds = [ood_config.ood_threshold]
    specificity = []
    sensitivity = []
    gmean = []

    transform = None
    for threshold in thresholds:
        if ood_config.evaluate_ood:
            print("====================================================")
            print("              Threshold: ", threshold)
            print("====================================================")

            network.ood_threshold = threshold

        #ds = data_load(root="/export/kiran/cityscapes/", split="val", transform=transform)/home/kumarasw/kiran/cityscapes_coco/cityscapes_val_coco_val
        ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split, transform=transform)
        detector = AnomalyDetector(network, ds, transform, model_name)
        result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds,
                               collate_fn=panoptic_deep_lab_collate, batch_size=ood_config.batch_size,
                               evaluate_ood=ood_config.evaluate_ood, semantic_only=ood_config.semantic_only,
                               evaluate_anomoly = ood_config.evaluate_anomoly)

        print(result)
        specificity.append(result['semantic_seg']['sem_seg']['uSpecificity'])
        sensitivity.append(result['semantic_seg']['sem_seg']['uSensitivity'])
        gmean.append(result['semantic_seg']['sem_seg']['uGmean'])

    if ood_config.evaluate_ood:
        if len(thresholds) > 1:
            fig = plt.figure()
            plt.plot(thresholds, specificity, label="uSpecificity")
            plt.plot(thresholds, sensitivity, label="uSensitivity")
            plt.plot(thresholds, gmean, label="uGmean")
            plt.xlabel("Threshold")
            plt.ylabel("uGmean")
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

    # use detectron2 distributed args
    args = default_argument_parser().parse_args()
    # args from current file
    args.default_args = vars(parser.parse_args())
    print("Command Line Args:", args)

    evaluate(args)
