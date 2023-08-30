import argparse
import time
import os
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
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
from PIL import Image
import matplotlib.pyplot as plt
from panopticapi.utils import rgb2id, id2rgb
from cityscapesscripts.helpers.labels import id2label, labels, trainId2label

ckpt_path = ood_config.init_ckpt
config_file = ood_config.config_file



class AnomalyDetector():
    def __init__(self, model=None, num_classes=19, results_path=None, ood_threshold=None):
        self.network = model
        self.num_classes = num_classes
        self.results_path = results_path
        self.ood_threshold = ood_threshold

    def estimator_worker(self, image):
        output = []
        for data in image:
            image_id = data["image_id"]
            city = image_id.split("_")[0]

            #semantic_image = np.array(Image.open(os.path.join(self.results_path, image_id+"_leftImg8bit_sem.png")))
            panoptic_image = np.array(Image.open(os.path.join(self.results_path, image_id+"_leftImg8bit_panoptic.png")))
            sem_uncertainity = np.load(os.path.join(self.results_path, image_id+"_leftImg8bit_uncMap.npy"))
            pan_uncertainity = np.load(os.path.join(self.results_path, image_id+"_leftImg8bit_uncMap_pan.npy"))

            sem_uncertainity = 1 - sem_uncertainity
            panoptic = rgb2id(panoptic_image)
            #semantic = rgb2id(semantic_image)
            semantic = np.zeros((panoptic.shape[0], panoptic.shape[1]), dtype="uint8")
            panoptic_ids = np.unique(panoptic)
            for id in panoptic_ids:
                if id < 1000:
                    labelInfo = id2label[id]
                    semantic_id = id
                    semantic[panoptic==id] = labelInfo.trainId
                else:
                    semantic_id = id // 1000
                    instance_count = id % 1000
                    labelInfo = id2label[semantic_id]
                    semantic[panoptic == id] = labelInfo.trainId
                    panoptic[panoptic == id] = labelInfo.trainId * 1000 + instance_count

            # det void region to -1
            semantic[semantic == 255] = -1
            semantic[sem_uncertainity > self.ood_threshold] = 19

            panopticids_at_ood = np.unique(panoptic[semantic==19])
            instance_count = 1
            for id in panopticids_at_ood:
                ood_id = 19000 +  instance_count
                panoptic[np.logical_and(panoptic==id, semantic==19)] = ood_id
                instance_count += 1

            # set void region to -1
            panoptic[panoptic<-1] = -1

            '''plt.imshow(semantic)
            plt.show()
            plt.imshow(panoptic)
            plt.show()
            plt.imshow(sem_uncertainity)
            plt.show()'''



            out = {
                "sem_seg": torch.tensor(semantic),
                "panoptic_seg": (torch.tensor(panoptic), None),
                "anomaly_score": torch.tensor(sem_uncertainity)
            }

        #output = self.network(image)

        if ood_config.save_results:
            self.display_results(image, output)

        return [out]

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



    transform = None
    #thresholds = [0.02* i for i in range(0,4)]
    thresholds = [ood_config.ood_threshold]
    specificity = []
    sensitivity = []
    gmean = []

    ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                   transform=transform)

    for threshold in thresholds:
        if ood_config.evaluate_ood:
            print("====================================================")
            print("              Threshold: ", threshold)
            print("====================================================")


        detector = AnomalyDetector(None, results_path=ood_config.result_path, ood_threshold=ood_config.ood_threshold)
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
            plt.plot(thresholds, specificity,  label="uSpecificity")
            plt.plot(thresholds, sensitivity,  label="uSensitivity")
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
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        help='possible datasets for statistics; cityscapes')

    args = parser.parse_args()
    print("Command Line Args:", args)

    evaluate(args)
