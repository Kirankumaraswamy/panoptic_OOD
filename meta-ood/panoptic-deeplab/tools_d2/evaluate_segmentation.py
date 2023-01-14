import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import config_training_setup
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.model_utils import load_network
from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
import config as meta_ood_config
from sklearn.metrics import confusion_matrix  

from PIL import Image
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

def compute_iou(pred, gt):
    pred = np.argmax(pred.numpy(), axis=0)
    gt = gt.numpy().squeeze()
    ious = []
    #pred[np.where(gt == 255)] = 255

    labels= np.arange(0, 19, 1)

    y_pred = pred.flatten()
    y_true = gt.flatten()

    '''index = np.where(y_true != 255)
    y_pred = y_pred[index]
    y_true = y_true[index]'''
    current = confusion_matrix(y_true, y_pred, labels=labels)
    return current


def getIouScoreForLabel(label, confMatrix):

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [i for i in range(19) if not i==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom




def panoptic_deep_lab_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target)
    return data, target

def evaluation_routine(config):
    """Start OoD Evaluation"""
    print("START EVALUATION")
    params = config.params
    roots = config.roots
    dataset = config.dataset(split="val", cs_root=roots.cs_root, coco_root=roots.coco_root, cs_split="val", coco_split="val")
    start_epoch = params.training_starting_epoch
    start = time.time()

    # This parameter is needed for panoptic segmentation data loading
    dataset_cfg = None

    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    if roots.model_name == "Detectron_DeepLab":
        add_deeplab_config(cfg)
        cfg.merge_from_file(meta_ood_config.Detectron_DeepLab_Config)
    elif roots.model_name == "Detectron_Panoptic_DeepLab":
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(meta_ood_config.Detectron_PanopticDeepLab_Config)
        dataset_cfg = cfg

    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt, train=False, cfg=cfg)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename), train=False, cfg=cfg)

    transform = Compose([])
    """Perform one epoch of training"""
    trainloader = config.dataset('val', transform, roots.cs_root, roots.coco_root, params.ood_subsampling_factor, cs_split="val", coco_split="val", cfg=dataset_cfg, model="detectron")
    dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=True, collate_fn=panoptic_deep_lab_collate)
    #dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=True)
    i = 0
    loss = None
    mious = []
    iou_list = [[] for i in range(19)]
    print("Total images: ", len(dataloader))
    pred_list = []
    gt_list = []
    confusion_matrix = np.zeros(shape=(19, 19),dtype=np.ulonglong)

    for count, (x, target) in enumerate(dataloader):
        print("count: ", count)
        logits = network(x)
        #logits = network(x.cuda())
        logits = logits[0]["sem_seg"]
        pred = logits.detach().cpu()

        del logits
        matrix = compute_iou(pred, target)
        confusion_matrix = confusion_matrix + matrix
        torch.cuda.empty_cache()
        a = 2


        '''target[np.where(target == 255)] = 19

        prediction = torch.squeeze(logits).detach().cpu().numpy().argmax(axis=0).astype(np.uint8)
        target = torch.squeeze(target[0]).numpy().astype(np.uint8)'''


        '''gt_name = os.path.join("/home/kumarasw/kiran/cityscapes_coco/deep_lab_results", "gt_"+str(count)+".png")
        pred_name = os.path.join("/home/kumarasw/kiran/cityscapes_coco/deep_lab_results" , "pred_"+str(count)+".png")
        Image.fromarray(torch.squeeze(logits).detach().cpu().numpy().argmax(axis=0).astype(np.uint8)).save(pred_name)
        Image.fromarray(torch.squeeze(target[0]).numpy().astype(np.uint8)).save(gt_name)'''

        '''pred_list.append(pred_name)
        gt_list.append(gt_name)

        encoding_value = max(target.max(), prediction.max()).astype(np.int32) + 1
        encoded = (target.astype(np.int32) * encoding_value) + prediction

        values, cnt = np.unique(encoded, return_counts=True)

        for value, c in zip(values, cnt):
            pred_id = value % encoding_value
            gt_id = int((value - pred_id)/encoding_value)
            confusion_matrix[gt_id][pred_id] += c'''
        
        
        '''import matplotlib.pyplot as plt
        plt.imshow(x[0]["image"].permute(1, 2, 0).numpy())
        #plt.imshow(x[0].permute(1, 2, 0).numpy())
        plt.savefig("/home/kumarasw/kiran/cityscapes_coco/image.png")
        out = torch.squeeze(logits).detach().cpu().numpy().argmax(axis=0)
        plt.imshow(out)
        plt.savefig("/home/kumarasw/kiran/cityscapes_coco/prediction.png")

        plt.imshow(target)
        plt.savefig("/home/kumarasw/kiran/cityscapes_coco/gt.png")'''


    ious = []
    for i in range(19):
        ious.append(getIouScoreForLabel(i, confusion_matrix))

    #print(confusion_matrix)
    print(ious)
    print("Results: ", np.nanmean(np.array(ious)))




def main(args):
    """Perform evaluation"""
    config = config_training_setup(args.default_args)
    evaluation_routine(config)

if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--training_starting_epoch", nargs="?", type=int)
    parser.add_argument("-nepochs", "--num_training_epochs", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-lr", "--learning_rate", nargs="?", type=float)
    parser.add_argument("-crop", "--crop_size", nargs="?", type=int)

    # use detectron2 distributed args
    args = default_argument_parser().parse_args()
    #args from current file
    args.default_args = vars(parser.parse_args())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    #main(vars(parser.parse_args()))
