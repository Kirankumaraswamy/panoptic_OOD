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
from cityscapesscripts.helpers.labels import id2label, trainId2label
import matplotlib.pyplot as plt
import ood_config
import _init_paths
import d2
import cv2
from PIL import Image
import seaborn as sns
sns.set()
ckpt_path = ood_config.init_ckpt
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

        image_save_dir = "/home/mohan/kiran/ponoptic_OOD/new_approach/tools_d2/sigmoidnet_results/cityscapes"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        image_save_path = os.path.join(image_save_dir, image[0]["image_id"] + "_comb.png")

        fig = plt.figure(figsize=(20, 14))
        rows = 2
        columns = 3
        images = []
        img1 = np.array(image[0]["real_image"])
        img1 = img1[50:-50, 50:-50,:]
    
        img2 = output[0]["sem_seg"].detach().cpu().squeeze().numpy()
        img2 = img2[50:-50, 50:-50]
        img3 = output[0]["anomaly_score"].detach().cpu().squeeze().numpy()
        img3 = img3[50:-50, 50:-50]
        pan_img = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()
        pan_img = pan_img[50:-50, 50:-50]
        pan_format = np.zeros(img1.shape, dtype="uint8")


        # remove ego vehicle pixels
        sem_gt = image[0]["sem_gt"][50:-50, 50:-50]
        pan_img[np.where(sem_gt==1)] = -1
        pan_img[-50:,:] = -1
        segment_ids = np.unique(pan_img)
        for segmentId in segment_ids:
            if segmentId >= 1000:
                semanticId = segmentId // 1000
            else:
                semanticId = segmentId
            mask = np.where(pan_img == segmentId)
            labelInfo = trainId2label[semanticId]
            color = labelInfo.color
            
            if semanticId == 19:
                color = np.array([214, 137, 16])
                mask = pan_img == segmentId
                #color =[np.random.randint(0, 255),np.random.randint(0, 255), np.random.randint(0, 255)]
                #color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                #color = labelInfo.color
                pan_format[mask] = color
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                vert = np.sum(mask, axis=1)
                if len(np.nonzero(hor)[0]) == 0 or len(np.nonzero(vert)[0]) == 0:
                    continue
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                # add start width and height to accommodate the bounding box positions in original cityscapes size
                bbox = [int(x),  int(y), int(width), int(height)]

                pan_format[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + 4] = color
                pan_format[bbox[1]: bbox[1] + bbox[3], bbox[0] + bbox[2]: bbox[0] + bbox[2] + 4] = color
                pan_format[bbox[1]: bbox[1] + 4, bbox[0]: bbox[0] + bbox[2]] = color
                pan_format[bbox[1] + bbox[3]: bbox[1] + bbox[3] + 4, bbox[0]: bbox[0] + bbox[2]] = color
                
            else:
                pan_format[mask] = labelInfo.color
        img4 = pan_format

        img5 = output[0]["centre_score"].detach().cpu().squeeze().numpy()
        img6 = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()
        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)

        pan_gt = np.zeros(img1.shape, dtype="uint8")
        for segmentId in np.unique(sem_gt):
            mask = np.where(sem_gt == segmentId)
            labelInfo = id2label[segmentId]

            if segmentId == 50:
                color = np.array([214, 137, 16])
            else:
                color = labelInfo.color
            pan_gt[mask] = color

        '''colors = np.random.randint(0, 255, size=(img6.max() + 1, 3), dtype=np.uint8)
        color_image = colors[img6]'''
        alpha = 0.5
        overlay = cv2.addWeighted(img1, 1 - alpha, pan_format, alpha, 0)
        overlay_gt = cv2.addWeighted(img1, 1 - alpha, pan_gt, alpha, 0)
        images.append(overlay)
        for i in range(6):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        fig.tight_layout()
        #plt.show()
        #plt.savefig(image_save_path)

        fig = plt.figure(dpi=300)

        plt.imshow(img3)
        plt.axis('off')
        fig.tight_layout()
        plt.savefig(os.path.join(image_save_dir, image[0]["image_id"] + "_uncertainity.png"))

        a=1
        Image.fromarray(img1).save(os.path.join(image_save_dir, image[0]["image_id"] + "_rgb.png"))
        #Image.fromarray(img4).save(os.path.join(image_save_dir, image[0]["image_id"] + "_pan.png"))
        Image.fromarray(overlay).save(os.path.join(image_save_dir, image[0]["image_id"] + "_overlay.png"))
        Image.fromarray(overlay_gt).save(os.path.join(image_save_dir, image[0]["image_id"] + "gt.png"))
        a=1

def panoptic_deep_lab_collate(batch):
    data = [item for item in batch]
    return data


def evaluate(args):
    """Start OoD Evaluation"""
    print("START EVALUATION")

    net = get_net()

    '''net.sum_all_class_mean = np.load(ood_config.sum_all_class_mean_path, allow_pickle=True).item()
    net.sum_all_class_var = np.load(ood_config.sum_all_class_var_path, allow_pickle=True).item()
    net.correct_class_mean = np.load(ood_config.correct_class_mean_path, allow_pickle=True).item()
    net.correct_class_var = np.load(ood_config.correct_class_var_path, allow_pickle=True).item()
    '''
    
    if ood_config.evaluate_ood or ood_config.performance_with_ood:
        net.performance_with_ood = ood_config.performance_with_ood 
        net.evaluate_ood = ood_config.evaluate_ood
    
    net.read_instance_path = ood_config.read_instance_path

    transform = None
    #thresholds = [0.02* i for i in range(0,4)]
    thresholds = [ood_config.ood_threshold]
    #thresholds = [0.0, 0.2,0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    #thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    specificity = []
    sensitivity = []

    pq_in = []
    pq_out = []
    pod_q = []

    gmean = []

    ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                   transform=transform)
    net.read_instance_path = ood_config.read_instance_path
    for threshold in thresholds:
        if ood_config.evaluate_ood or ood_config.performance_with_ood:
            print("====================================================")
            print("              Threshold: ", threshold)
            print("====================================================")


            net.ood_threshold = threshold
        detector = AnomalyDetector(net)
        result = data_evaluate(estimator=detector.estimator_worker, evaluation_dataset=ds,
                               collate_fn=panoptic_deep_lab_collate, batch_size=ood_config.batch_size,
                               evaluate_ood=ood_config.evaluate_ood, semantic_only=ood_config.semantic_only,
                               evaluate_anomoly = ood_config.evaluate_anomoly)

        print(result)
        pq_in.append(result['panotic_seg']['panoptic_seg']['PQ-in'])
        pq_out.append(result['panotic_seg']['panoptic_seg']['PQ-out'])
        pod_q.append(result['panotic_seg']['panoptic_seg']['POD-Q'])


        specificity.append(result['semantic_seg']['sem_seg']['uSpecificity'])
        sensitivity.append(result['semantic_seg']['sem_seg']['uSensitivity'])
        gmean.append(result['semantic_seg']['sem_seg']['uGmean'])

    if ood_config.evaluate_ood:
        if len(thresholds) > 1:
            
            default_x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            fig = plt.figure()
            plt.plot(thresholds, specificity,  label="uSpecificity")
            plt.plot(thresholds, sensitivity,  label="uSensitivity")
            plt.plot(thresholds, gmean, label="G-Mean")
            plt.xlabel("Threshold")
            plt.ylabel("Performance scaled to 1")
            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()
            plt.show()
            fig.savefig("./bdd_sigmoid_stage_1_sensitivity_vs_specificity_val.png",dpi=200)

            fig = plt.figure()
            plt.plot(thresholds, pq_in, label="PQ-in")
            plt.plot(thresholds, pq_out, label="PQ-out")
            plt.plot(thresholds, pod_q, label="POD-Q")
            plt.xlabel("Threshold")
            plt.ylabel("Performance in %")

            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()

            fig.savefig("./bdd_sigmoid_stage_1_podq_threshold_val.png", dpi=200)

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
