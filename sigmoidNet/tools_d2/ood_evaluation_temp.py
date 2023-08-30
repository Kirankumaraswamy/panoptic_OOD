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
from panopticapi.utils import rgb2id
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
from scipy.stats import entropy
import numpy as np
from cityscapesscripts.helpers.labels import trainId2label, id2label
import matplotlib.pyplot as plt
import ood_config
import cv2
import _init_paths
import d2
from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

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

        image_save_dir = "/home/mohan/kiran/ponoptic_OOD/new_approach/tools_d2/sigmoidnet_results/sigmoid_stage_1/bdd"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        image_save_path = os.path.join(image_save_dir, image[0]["image_id"] + "_comb.png")

        fig = plt.figure(figsize=(20, 14))
        rows = 2
        columns = 3
        images = []

        cropsize = 10
        img1 = np.array(image[0]["real_image"])
        img1 = img1[cropsize:-cropsize,cropsize:-cropsize]
        img2 = output[0]["sem_seg"].detach().cpu().squeeze().numpy()
        img2 = img2[cropsize:-cropsize,cropsize:-cropsize]
        img3 = output[0]["anomaly_score"].detach().cpu().squeeze().numpy()
        img3 = img3[cropsize:-cropsize,cropsize:-cropsize]
        pan_img = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()
        pan_img = pan_img[cropsize:-cropsize,cropsize:-cropsize]

        sem_gt = image[0]["sem_gt"]
        sem_gt = sem_gt[cropsize:-cropsize,cropsize:-cropsize]

        #pan_img = rgb2id(image[0]["pan_gt"])[cropsize:-cropsize,cropsize:-cropsize]
        pan_img[sem_gt==1] = -1
        #pan_img[-100:100,100:-100] = -1
        segment_ids = np.unique(pan_img)
        pan_format = np.zeros(img1.shape, dtype="uint8")
        pan_pred = pan_img.copy()
        #sem_score =  F.softmax(output[0]["sem_score"], dim=0).detach().cpu().squeeze().numpy().max(axis=0)
        sem_score = output[0]["sem_score"].detach().cpu().squeeze().numpy().max(axis=0)
        sem_score = sem_score[cropsize:-cropsize,cropsize:-cropsize]
        for segmentId in segment_ids:
            if segmentId >= 1000:
                semanticId = segmentId // 1000
            else:
                semanticId = segmentId
            mask = np.where(pan_img == segmentId)
            labelInfo = trainId2label[semanticId]
            pan_format[mask] = labelInfo.color

            if labelInfo.hasInstances or semanticId == 19:
                mask = pan_img == segmentId
                mask1 = np.logical_and(mask, sem_gt==1)
                match = np.sum(mask1)
                #color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                color = labelInfo.color
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

                pan_format[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + 5] = color
                pan_format[bbox[1]: bbox[1] + bbox[3], bbox[0] + bbox[2]: bbox[0] + bbox[2] + 5] = color
                pan_format[bbox[1]: bbox[1] + 5, bbox[0]: bbox[0] + bbox[2]] = color
                pan_format[bbox[1] + bbox[3]: bbox[1] + bbox[3] + 5, bbox[0]: bbox[0] + bbox[2]] = color

                '''x1, y1, x2, y2 = [int(x), int(y), int(x)+int(width), int(y)+int(height) ]
                score = sem_score[mask].mean()
                label = labelInfo.name + " "+ str(round(score * 100, 1)) + "%"
                # Position the label on top of the bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX
                textsize = cv2.getTextSize(label, font, 1, 2)[0]
                text_x = x1 + int((x2 - x1) / 2) - int(textsize[0] / 2)
                #text_y = y1 + int((y2 - y1) / 2) + int(textsize[1] / 2)
                text_x = x1
                text_y = y1 -10

                # Overlay the label on top of the image
                #img = cv2.imread('object_detection_image.jpg')
                #cv2.rectangle(img1, (x1, y1), (x2, y2), color, 2)
                #cv2.putText(img1, label, (text_x, text_y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)'''
            else:
                pan_pred[mask] = 0


        img4 = pan_format

        '''img5 = output[0]["centre_score"].detach().cpu().squeeze().numpy()
        img6 = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()
        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)'''''

        pan_gt = np.zeros(img1.shape, dtype="uint8")
        for segmentId in np.unique(sem_gt):
            mask = np.where(sem_gt == segmentId)
            labelInfo = id2label[segmentId]
            pan_gt[mask] = labelInfo.color

        '''colors = np.random.randint(0, 255, size=(img6.max() + 1, 3), dtype=np.uint8)
        color_image = colors[img6]'''
        alpha = 0.5
        overlay = cv2.addWeighted(img1, 1 - alpha, pan_format, alpha, 0)
        overlay_gt = cv2.addWeighted(img1, 1 - alpha, pan_gt, alpha, 0)


        contours = find_boundaries(pan_pred, mode="outer", background=0).astype(np.uint8) * 255
        contours = dilation(contours)

        contours = np.expand_dims(contours, -1).repeat(4, -1)
        contours_img = Image.fromarray(contours, mode="RGBA")
        out = Image.alpha_composite(Image.fromarray(overlay).convert(mode="RGBA"), contours_img)
        '''out = np.array(out.convert('RGB'))
        for segmentId in segment_ids:
            if segmentId >= 1000:
                semanticId = segmentId // 1000
            else:
                semanticId = segmentId
            mask = np.where(pan_img == segmentId)
            labelInfo = trainId2label[semanticId]
            pan_format[mask] = labelInfo.color

            if labelInfo.hasInstances or semanticId == 19:
                mask = pan_img == segmentId
                mask1 = np.logical_and(mask, sem_gt==1)
                match = np.sum(mask1)
                #color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                color = labelInfo.color
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

                out[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + 5] = color
                out[bbox[1]: bbox[1] + bbox[3], bbox[0] + bbox[2]: bbox[0] + bbox[2] + 5] = color
                out[bbox[1]: bbox[1] + 5, bbox[0]: bbox[0] + bbox[2]] = color
                out[bbox[1] + bbox[3]: bbox[1] + bbox[3] + 5, bbox[0]: bbox[0] + bbox[2]] = color

                x1, y1, x2, y2 = [int(x), int(y), int(x)+int(width), int(y)+int(height) ]
                score = sem_score[mask].mean()
                label = labelInfo.name + " "+ str(round(score * 100, 1)) + "%"
                # Position the label on top of the bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX
                textsize = cv2.getTextSize(label, font, 1, 2)[0]
                text_x = x1 + int((x2 - x1) / 2) - int(textsize[0] / 2)
                #text_y = y1 + int((y2 - y1) / 2) + int(textsize[1] / 2)
                text_x = x1
                text_y = y1 -10

                # Overlay the label on top of the image
                #img = cv2.imread('object_detection_image.jpg')
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)'''

        #Image.fromarray(img1).save(os.path.join(image_save_dir, image[0]["image_id"] + "_bounding_box.png"))
        out.save(os.path.join(image_save_dir, image[0]["image_id"] + "_overlay.png"))


        '''images.append(overlay)
        for i in range(6):
            fig.add_subplot(rows, columns, i + 1)
            #plt.imshow(images[i])
            plt.axis('off')'''

        fig.tight_layout()
        #plt.show()
        #plt.savefig(image_save_path)

        '''fig = plt.figure(dpi=300)
        plt.imshow(img1)
        plt.axis('off')
        fig.tight_layout()
        plt.show()

        fig = plt.figure(dpi=300)'''

        plt.imshow(img3)
        plt.axis('off')
        fig.tight_layout()
        #plt.savefig(os.path.join(image_save_dir, image[0]["image_id"] + "_uncertainity.png"))


        a=1
        #Image.fromarray(img1).save(os.path.join(image_save_dir, image[0]["image_id"] + "_rgb.png"))
        #Image.fromarray(img4).save(os.path.join(image_save_dir, image[0]["image_id"] + "_pan.png"))
        #Image.fromarray(overlay).save(os.path.join(image_save_dir, image[0]["image_id"] + "_overlay.png"))
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
    net.correct_class_var = np.load(ood_config.correct_class_var_path, allow_pickle=True).item()'''


    '''net.class_mean_neg = np.load(ood_config.class_mean_neg, allow_pickle=True).item()
    net.class_var_neg = np.load(ood_config.class_var_neg, allow_pickle=True).item()

    net.class_mean_sum_all = np.load(ood_config.class_mean_sum_all, allow_pickle=True).item()
    net.class_var_sum_all = np.load(ood_config.class_var_sum_all, allow_pickle=True).item()

    net.class_mean_sum_all_neg = np.load(ood_config.class_mean_sum_all_neg, allow_pickle=True).item()
    net.class_var_sum_all_neg = np.load(ood_config.class_var_sum_all_neg, allow_pickle=True).item()'''

    net.evaluate_ood = ood_config.evaluate_ood
    net.ood_threshold_val = ood_config.ood_threshold

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


            net.ood_threshold = threshold
        detector = AnomalyDetector(net)
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

