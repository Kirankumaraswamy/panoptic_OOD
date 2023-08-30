import argparse
import time
import os, sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from dataset.cityscapes_ood import CityscapesOOD
import ood_config as config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from torchvision.transforms import Compose, RandomHorizontalFlip, Normalize, ToTensor
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.projects.deeplab import build_lr_scheduler
import warnings

warnings.filterwarnings('ignore')
from panoptic_evaluation.evaluation import data_load, data_evaluate
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data.distributed import DistributedSampler
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
import detectron2.data.transforms as T
import matplotlib.pyplot as plt
from scipy.stats import entropy


import _init_paths
import d2


def panoptic_deep_lab_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.stack(target)
    return data, target


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]

    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

    augs.append(T.RandomFlip())
    return []


def panoptic_deep_lab_val_collate(batch):
    data = [item for item in batch]
    return data


def eval_metric_model(network):
    loss = None
    sys.stdout = open(os.devnull, 'w')
    sys.tracebacklimit = 0
    ds = data_load(root=config.cityscapes_eval_path, split="val", transform=None)
    result = data_evaluate(estimator=network, evaluation_dataset=ds,
                           collate_fn=panoptic_deep_lab_val_collate, semantic_only=False, evaluate_ood=False)
    sys.stdout = sys.__stdout__
    return result


def calculate_statistics(args, network, dataset_cfg):
    print("Evaluate statistics: ", comm.get_local_rank())
    ckpt_path = config.init_ckpt

    checkpointer = DetectionCheckpointer(
        network
    )

    if ckpt_path is not None:
        print("Checkpoint file:", ckpt_path)
        checkpointer.resume_or_load(
            ckpt_path, resume=False
        )
    else:
        raise ("Check point file cannot be None.")

    dataset_train = CityscapesOOD(root=config.ood_dataset_path, split=config.ood_split, cfg=dataset_cfg,
                                  transform=build_sem_seg_train_aug(dataset_cfg))

    start = time.time()
    # network = torch.nn.DataParallel(network).cuda()
    network = network.cuda()

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=False,
                                  collate_fn=panoptic_deep_lab_collate, num_workers=0)
    network.eval()

    pred_list = None
    target_list = None
    max_class_mean = {}
    print("Calculating statistics...")

    #statistics_file_name = config.statistics_file_name

    for i, (x, target) in enumerate(dataloader_train):
        # print("Train : len of data loader: ", len(dataloader_train), comm.get_rank())

        '''plt.imshow(x[0]["image"].permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(torch.squeeze(x[0]["sem_seg"]).detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.squeeze(x[0]["center"]).detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.squeeze(x[0]["center_weights"]).detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.squeeze(x[0]["ood_mask"]).detach().cpu().numpy())
        plt.show()'''
        output_list = []
        with torch.no_grad():
            outputs = network(x)

        '''sem = outputs.argmax(dim=0)
        mask = torch.sum((outputs > 0.5) * 1.0, axis=0)
        result = torch.ones_like(sem) * 19
        result[mask == 1] = sem[mask == 1]

        ood_mask = torch.ones_like(sem)
        ood_mask[mask == 1] = 0

        plt.imshow(torch.squeeze(sem).detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.squeeze(ood_mask).detach().cpu().numpy())
        plt.show()'''

        for index, output in enumerate(outputs):
            if pred_list is None:
                pred_list = output['sem_score'].cpu().unsqueeze(dim=0)
                target_list = x[index]["sem_seg"].unsqueeze(dim=0)
            else:
                pred_list = torch.cat((pred_list, output['sem_score'].cpu().unsqueeze(dim=0)), 0)
                target_list = torch.cat((target_list, x[index]["sem_seg"].unsqueeze(dim=0)), 0)

        del outputs, output
        torch.cuda.empty_cache()
        print("batch: ", i)

        if i % 200 == 199 or i == len(dataloader_train) - 1:
            break
    
    print(pred_list.size(), pred_list.min(), pred_list.max())
    #pred_list = F.softmax(pred_list, dim=1)
    print(pred_list.size(), pred_list.min(), pred_list.max())
    pred_list = pred_list.permute(0, 2, 3, 1)
    pred_score, prediction = pred_list.max(3)
    print(pred_score.size(), pred_score.min(), pred_score.max())

    class_sum_all_probabilities = []
    correct_class_probabilities = []
    non_class_probabilities = []
    mean_sum_all_dict, var_sum_all_dict = {}, {}
    mean_sum_non_class_dict, var_sum_non_class_dict = {}, {}
    mean_class_dict, var_class_dict = {}, {}

    mean_class = []
    std_class = []
    mean_non_class = []
    std_non_class = []

    ood_mean_class = []
    ood_std_class = []
    ood_mean_non_class = []
    ood_std_non_class = []

    mean_sum_all = []
    std_sum_all = []

    diff_mean_dict = {}
    diff_var_dict = {}
    diff_mean = []
    diff_std = []

    ood_diff_mean_dict = {}
    ood_diff_var_dict = {}
    ood_diff_mean = []
    ood_diff_std = []

    n_in_pixels = []
    n_out_pixels = []

    cls_list = [i for i in range(19)]

    for c in range(19):
        if c in cls_list:
            correct_pred = target_list == prediction
            class_pred = prediction == c

            correct_class_pred = torch.logical_and(correct_pred, class_pred)
            print("class: ", c, ", pixels considered:  ", correct_class_pred.sum())
            n_in_pixels.append(((correct_class_pred.sum()/((target_list==c).sum()+0.000001))*100).item())

            sum_all_probabilities = pred_list[correct_class_pred].sum(axis=1)
            mean_sum_all_dict[c] = sum_all_probabilities.mean(dim=0).item()
            var_sum_all_dict[c] = sum_all_probabilities.var(dim=0).item()

            mean_sum_all.append(sum_all_probabilities.mean(dim=0).item())
            std_sum_all.append(sum_all_probabilities.std(dim=0).item())

            right_probabilities = pred_list[correct_class_pred][:, c]
            mean_class_dict[c] = right_probabilities.mean().item()
            var_class_dict[c] = right_probabilities.var().item()

            mean_class.append(right_probabilities.mean(dim=0).item())
            std_class.append(right_probabilities.std(dim=0).item())

            sum_non_class_probabilities = sum_all_probabilities - right_probabilities
            mean_sum_non_class_dict[c] = sum_non_class_probabilities.mean().item()
            var_sum_non_class_dict[c] = sum_non_class_probabilities.var()

            mean_non_class.append(sum_non_class_probabilities.mean(dim=0).item())
            std_non_class.append(sum_non_class_probabilities.std(dim=0).item())

            diff = right_probabilities - sum_non_class_probabilities
            diff_mean_dict[c] = diff.mean().item()
            diff_var_dict = diff.var().item()
            diff_mean.append(diff.mean().item())
            diff_std.append(diff.std().item())

    out_list = target_list == 19

    ood_mean_sum_all = []
    ood_std_sum_all = []

    for c in range(19):
        if c in cls_list:
            class_pred = prediction == c

            out_class_pred = torch.logical_and(out_list, class_pred)
            n_out_pixels.append(out_class_pred.sum())
            
            ood_mean_class.append(pred_score[out_class_pred].mean().item())
            ood_std_class.append(pred_score[out_class_pred].std().item())

            sum_all_probabilities = pred_list[out_class_pred].sum(axis=1)

            sum_non_class_probabilities = sum_all_probabilities - pred_score[out_class_pred]

            ood_mean_non_class.append(sum_non_class_probabilities.mean().item())
            ood_std_non_class.append(sum_non_class_probabilities.std().item())

            ood_mean_sum_all.append(sum_all_probabilities.mean().item())
            ood_std_sum_all.append(sum_all_probabilities.std().item())

            diff = pred_score[out_class_pred] - sum_non_class_probabilities
            ood_diff_mean_dict[c] = diff.mean().item()
            ood_diff_var_dict = diff.var().item()
            ood_diff_mean.append(diff.mean().item())
            ood_diff_std.append(diff.std().item())

    print(f"sum all class mean: {mean_sum_all_dict}")
    print(f"sum all class var: {var_sum_all_dict}")
    print("======================")
    print(f"correct class mean: {mean_class_dict}")
    print(f"correct class var: {var_class_dict}")
    print("======================")
    print(f"sum all non class mean: {mean_sum_non_class_dict}")
    print(f"sum all non class var: {var_sum_non_class_dict}")
    print("======================")
    print(ood_mean_class)
    print(ood_std_class)
    '''np.save(f'./stats/diff_{statistics_file_name}_mean.npy', diff_mean_dict)
    np.save(f'./stats/diff_{statistics_file_name}_var.npy', diff_var_dict)
    np.save(f'./stats/correct_class_{statistics_file_name}_mean.npy', mean_class_dict)
    np.save(f'./stats/correct_class_{statistics_file_name}_var.npy', var_class_dict)'''

    name = "sigmoid_stage_1"
    x = [i for i in range(len(cls_list))]
    plt.rcParams.update({'font.size': 16})
    default_x_ticks = [int(i) for i in range(19)]
    default_x_ticks_val= ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
                          "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    #default_x_ticks_val = [int(i) + 1 for i in range(19)]
    fig = plt.figure("In distribution and ood " + name, figsize=(12, 8))
    plt.errorbar(x, mean_class, std_class, elinewidth=3, capsize=5, linestyle='None', marker='^', markersize=10, label="in-distribution")
    # plt.errorbar(x, mean_non_class, std_non_class, linestyle='None', marker='^')

    '''x = [i + 0.1 for i in range(len(cls_list))]
    plt.errorbar(x, mean_non_class, std_non_class, linestyle='None', marker='^')
    '''
    x = [i + 0.2 for i in range(len(cls_list))]
    plt.errorbar(x, ood_mean_class, ood_std_class, elinewidth=3, capsize=5, linestyle='None', marker='^', markersize=10, label="out-of-distribution")
    # plt.errorbar(x, ood_mean_non_class, ood_std_non_class, linestyle='None', marker='^')

    '''x = [i + 0.3 for i in range(len(cls_list))]
    plt.errorbar(x, ood_mean_non_class, ood_std_non_class, linestyle='None', marker='^')
    '''
    defaul_y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.xticks(default_x_ticks, default_x_ticks_val, ha='center', rotation="vertical")
    plt.yticks(defaul_y_ticks, defaul_y_ticks)
    #plt.title("Probability distribution "+name)
    plt.xlabel("In-distribution classes", fontweight='bold')
    plt.ylabel("Sigmoid prediction probability", fontweight='bold')
    plt.tight_layout()
    plt.legend()

    #plt.show()
    fig.savefig(f"./correct_class_{name}.png", dpi=200)

    fig = plt.figure("Performance "+ name, figsize=(12, 8))

    x = [i for i in range(len(cls_list))]
    plt.bar(x, n_in_pixels)

    total = 0
    for  v in n_out_pixels:
        total += v.item()
    for i in range(len(x)):
        plt.text(i, n_in_pixels[i], "{:.1f} %".format(n_in_pixels[i]) , ha='center', rotation=15, fontsize="x-small")
    plt.xticks(default_x_ticks, default_x_ticks_val, ha='center', rotation="vertical")
    #plt.title("Semantic segmentation performance "+name)
    plt.ylabel("Correct prediction in percentatge", fontweight='bold')
    plt.xlabel("In-distribution classes", fontweight='bold')
    plt.tight_layout()
    #plt.show()
    fig.savefig(f"./performance_{name}.png", dpi=200)

    
    fig = plt.figure("OOD "+ name, figsize=(12, 8))

    x = [i for i in range(len(cls_list))]
    plt.bar(x, n_out_pixels)

    total = 0
    for  v in n_out_pixels:
        total += v.item()
    for i in range(len(x)):
        plt.text(i, n_out_pixels[i], "{:.1f} %".format((n_out_pixels[i].item()/total) * 100) , ha='center', rotation=15, fontsize="x-small")
    plt.xticks(default_x_ticks, default_x_ticks_val, ha='center', rotation="vertical")
    #plt.title("OOD pixels prediction "+name)
    plt.ylabel("Pixel prediction count", fontweight='bold')
    plt.xlabel("In-distribution classes", fontweight='bold')
    plt.tight_layout()
    #plt.show()
    fig.savefig(f"./pixel_count_{name}.png", dpi=200)

    print("=====================================")
    print(n_in_pixels)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main(args):
    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config.config_file)
    cfg.freeze()
    default_setup(
        cfg, args
    )

    network = build_model(cfg)
    print("Model:\n{}".format(network))

    """calculate_statistics"""
    calculate_statistics(args, network, cfg)


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
    # args from current file
    args.default_args = vars(parser.parse_args())
    args.dist_url = 'tcp://127.0.0.1:64486'
    print("Command Line Args:", args)

    main(args)

