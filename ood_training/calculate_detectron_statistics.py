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
import config
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
    ckpt_path = config.ckpt_path

    checkpointer = DetectionCheckpointer(
        network
    )

    if ckpt_path is not None:
        print("Checkpoint file:", ckpt_path)
        checkpointer.resume_or_load(
            ckpt_path, resume=False
        )
    else:
        raise("Check point file cannot be None.")

    dataset_train = CityscapesOOD(root=config.cityscapes_ood_path, split=config.split, cfg=dataset_cfg,
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

    statistics_file_name = config.statistics_file_name

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
        with torch.no_grad():
            output = network(x)

        outputs = output[0]['sem_score']

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

        outputs = outputs.unsqueeze(dim=0)
        if pred_list is None:
            pred_list = outputs.data.cpu()
            target_list = x[0]["sem_seg"].unsqueeze(dim=0)
        else:
            pred_list = torch.cat((pred_list, outputs.cpu()), 0)
            target_list = torch.cat((target_list, x[0]["sem_seg"].unsqueeze(dim=0)), 0)

        del outputs, output
        torch.cuda.empty_cache()
        print("batch: ", i)

        if i % 200 == 199 or i == len(dataloader_train) - 1:
            break

    pred_list = pred_list.permute(0, 2, 3, 1)
    pred_score, prediction = pred_list.max(3)

    class_sum_all_probabilities = []
    correct_class_probabilities = []
    non_class_probabilities = []
    mean_sum_all_dict, var_sum_all_dict = {}, {}
    mean_sum_non_class_dict, var_sum_non_class_dict = {}, {}
    mean_class_dict, var_class_dict = {}, {}


    for c in range(19):
        correct_pred = target_list == prediction
        class_pred = prediction == c

        correct_class_pred = torch.logical_and(correct_pred, class_pred)
        print("class: ", c, ", pixels considered:  ", correct_class_pred.sum())

        sum_all_probabilities = pred_list[correct_class_pred].sum(axis=1)
        class_sum_all_probabilities.append(sum_all_probabilities)
        mean_sum_all_dict[c] = class_sum_all_probabilities[c].mean(dim=0).item()
        var_sum_all_dict[c] = class_sum_all_probabilities[c].var(dim=0).item()

        right_probabilities = pred_list[correct_class_pred][:, c]
        correct_class_probabilities.append(right_probabilities)
        mean_class_dict[c] = correct_class_probabilities[c].mean().item()
        var_class_dict[c] = correct_class_probabilities[c].var().item()

        sum_non_class_probabilities = sum_all_probabilities - right_probabilities
        non_class_probabilities.append(sum_non_class_probabilities)
        mean_sum_non_class_dict[c] = non_class_probabilities[c].mean().item()
        var_sum_non_class_dict[c] = non_class_probabilities[c].var().item()


    print(f"sum all class mean: {mean_sum_all_dict}")
    print(f"sum all class var: {var_sum_all_dict}")
    print("======================")
    print(f"correct class mean: {mean_class_dict}")
    print(f"correct class var: {var_class_dict}")
    print("======================")
    print(f"sum all non class mean: {mean_sum_non_class_dict}")
    print(f"sum all non class var: {var_sum_non_class_dict}")
    print("======================")

    np.save(f'./stats/sum_all_{statistics_file_name}_mean.npy', mean_sum_all_dict)
    np.save(f'./stats/sum_all_{statistics_file_name}_var.npy', var_sum_all_dict)
    np.save(f'./stats/correct_class_{statistics_file_name}_mean.npy', mean_sum_all_dict)
    np.save(f'./stats/correct_class_{statistics_file_name}_var.npy', var_sum_all_dict)
    np.save(f'./stats/sum_non_class_{statistics_file_name}_mean.npy', mean_sum_all_dict)
    np.save(f'./stats/sum_non_class_{statistics_file_name}_var.npy', var_sum_all_dict)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main(args):
    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config.detectron_config_file_path)
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
