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

model_name = "Detectron_Panoptic_DeepLab"
# ckpt_path = "/home/kumarasw/Thesis/Standardized-max-logits/pretrained/deeplab_model_final_a8a355.pkl"
ckpt_path = "/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"


Detectron_PanopticDeepLab_Config = "/home/kumarasw/Thesis/Standardized-max-logits/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
Detectron_DeepLab_Config = "/home/kumarasw/Thesis/Standardized-max-logits/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"


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
            cfg.merge_from_file(Detectron_DeepLab_Config)
        elif model_name == "Detectron_Panoptic_DeepLab":
            add_panoptic_deeplab_config(cfg)
            cfg.merge_from_file(Detectron_PanopticDeepLab_Config)
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

        if not "anomaly_score" in output[0].keys():
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


        '''import matplotlib.pyplot as plt
        plt.imshow(image[0]["image"].cpu().permute(1, 2, 0).numpy())
        plt.show()
        plt.savefig("/home/kumarasw/Thesis/image.png")
        plt.imshow(prediction)
        plt.show()
        plt.savefig("/home/kumarasw/Thesis/mask.png")
        plt.imshow(anomaly_score)
        plt.show()
        plt.savefig("/home/kumarasw/Thesis/anomoly.png")'''

        #return anomaly_score.cpu()
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
    os.chdir("/home/kumarasw/Thesis/Standardized-max-logits")
    net = get_net()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    class_mean = np.load(f'./stats/{args.dataset}_{model_name}_mean.npy', allow_pickle=True)
    class_var = np.load(f'./stats/{args.dataset}_{model_name}_var.npy', allow_pickle=True)

    transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)])
    ds = data_load(root="/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes_ood", split="val",
                   transform=transform)
    net.class_mean = class_mean.item()
    net.class_var=class_var.item()
    detector = AnomalyDetector(net, mean_std, class_mean=class_mean.item(), class_var=class_var.item(), model_name=model_name)
    result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds,
                           collate_fn=panoptic_deep_lab_collate, semantic_only=False)
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
