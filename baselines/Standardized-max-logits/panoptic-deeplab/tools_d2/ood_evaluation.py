import argparse
import time
import os
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
import config as meta_ood_config
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from config import cfg, assert_and_infer_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser
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
import seaborn as sns
sns.set()
import _init_paths
import d2

model_name = ood_config.model_name
ckpt_path = ood_config.init_ckpt
config_file = ood_config.config_file


def get_net():
    """
    Main Function
    """

    train = False
    
    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)

    if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
        cfg = get_cfg()
        if model_name == "Detectron_DeepLab":
            add_deeplab_config(cfg)
            cfg.merge_from_file(config_file)
        elif model_name == "Detectron_Panoptic_DeepLab":
            add_panoptic_deeplab_config(cfg)
            cfg.merge_from_file(config_file)
        network = build_model(cfg)
        # network = torch.nn.DataParallel(network).cuda()
    else:
        print("\nModel is not known")
        exit()

    if ckpt_path is not None:
        if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
            DetectionCheckpointer(network).resume_or_load(
                ckpt_path, resume=False
            )
        else:
            # network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
            network.load_state_dict(torch.load(ckpt_path)['model'], strict=False)
    seg_net = network.cuda()
    if train:
        print("... ok")
        seg_net.train()
    else:
        print("... ok")
        seg_net.eval()

    return seg_net


class AnomalyDetector():
    def __init__(self, model=None, mean_std=None, num_classes=19, class_mean=None, class_var=None, model_name=None):
        self.network = model
        self.model_name = model_name
        self.mean_std = mean_std
        self.class_mean = class_mean
        self.class_var = class_var
        self.num_classes = num_classes

    def estimator_worker(self, image):
        image = self.preprocess_image(image)
        output = self.network(image)

        if False and ood_config.evaluate_ood and not "anomaly_score" in output[0].keys():
            main_out = torch.unsqueeze(output[0]['sem_seg'], dim=0)

            if self.class_mean is None or self.class_var is None:
                raise Exception("Class mean and var are not set!")
            anomaly_score, prediction = main_out.detach().cpu().max(1)
            for c in range(self.num_classes):
                anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - self.class_mean[c]) / np.sqrt(self.class_var[c]),
                                            anomaly_score)

            del main_out
            anomaly_score = anomaly_score * -1

            prediction = torch.squeeze(prediction).cpu().numpy()
            anomaly_score = torch.squeeze(anomaly_score).cpu().numpy()
            prediction[np.where(anomaly_score > 2)] = 19

            output[0]['sem_seg'] = torch.tensor(prediction)
            output[0]['anomaly_score'] = torch.tensor(anomaly_score)
        
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
    #os.chdir("/home/kumarasw/Thesis/Standardized-max-logits")
    net = get_net()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    net.performance_with_ood = ood_config.performance_with_ood
    if ood_config.evaluate_ood or ood_config.performance_with_ood:
        class_mean = np.load(f'./stats/{args.dataset}_{model_name}_mean.npy', allow_pickle=True)
        class_var = np.load(f'./stats/{args.dataset}_{model_name}_var.npy', allow_pickle=True)
        
        class_mean = np.load(ood_config.class_mean, allow_pickle=True)
        class_var = np.load(ood_config.class_var, allow_pickle=True)
        print(class_mean.item())
        print(class_var.item())
        
        net.class_mean = class_mean.item()
        net.class_var = class_var.item()
        net.evaluate_ood = ood_config.evaluate_ood

        print(class_mean)

    net.read_instance_path = ood_config.read_instance_path

    #transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)])
    transform = None
    thresholds = [0.0,  0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.5, 0.6, 0.7, 0.9, 1.0]

    thresholds = [ood_config.ood_threshold]
    specificity = []
    sensitivity = []
    gmean = []

    pq_in = []
    pq_out = []
    pod_q = []

    ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                   transform=transform)

    for threshold in thresholds:
        if ood_config.evaluate_ood or ood_config.performance_with_ood:
            print("====================================================")
            print("              Threshold: ", threshold)
            print("====================================================")
            net.ood_threshold = threshold
        detector = AnomalyDetector(net, mean_std, model_name=model_name)
        result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds, batch_size=ood_config.batch_size,
                               collate_fn=panoptic_deep_lab_collate, evaluate_ood=ood_config.evaluate_ood, semantic_only=ood_config.semantic_only,
                               evaluate_anomoly = ood_config.evaluate_anomoly)

        print(result)
        specificity.append(result['semantic_seg']['sem_seg']['uSpecificity'])
        sensitivity.append(result['semantic_seg']['sem_seg']['uSensitivity'])
        gmean.append(result['semantic_seg']['sem_seg']['uGmean'])
        
        pq_in.append(result['panotic_seg']['panoptic_seg']['PQ-in'])
        pq_out.append(result['panotic_seg']['panoptic_seg']['PQ-out'])
        pod_q.append(result['panotic_seg']['panoptic_seg']['POD-Q'])

    if ood_config.evaluate_ood:
        if len(thresholds) > 1:
            default_x_ticks = [0.0,  0.1,  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            fig = plt.figure()
            plt.plot(thresholds, specificity,  label="uSpecificity")
            plt.plot(thresholds, sensitivity,  label="uSensitivity")
            plt.plot(thresholds, gmean, label="G-Mean")
            plt.xlabel("Threshold")
            plt.ylabel("Performance scaled to 1")
            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()
            plt.show()
            fig.savefig("./max_logit_sensitivity_vs_specificity_val.png",dpi=200)

            fig = plt.figure()
            plt.plot(thresholds, pq_in, label="PQ-in")
            plt.plot(thresholds, pq_out, label="PQ-out")
            plt.plot(thresholds, pod_q, label="POD-Q")
            plt.xlabel("Threshold")
            plt.ylabel("Performance in %")

            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()

            fig.savefig("./max_logit_podq_threshold_val.png", dpi=200)

        print("Thresholds: ", thresholds)
        print("Gmean: ", gmean)
        print('Usensitivity: ', sensitivity)
        print("Uspecivicity: ", specificity)

        print("============================================")
        print("PQ-in", pq_in)
        print("PQ-out", pq_out)
        print("POD-Q", pod_q)


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
