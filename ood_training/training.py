import argparse
import time
import os, sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
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
from detectron2.solver import build_lr_scheduler, build_optimizer
import warnings
warnings.filterwarnings('ignore')
from panoptic_evaluation.evaluation import data_load, data_evaluate
import matplotlib.pyplot as plt
import tqdm

def panoptic_deep_lab_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.stack(target)
    return data, target

def train_model(network, dataloader_train, optimizer, scheduler, epoch=None):
    loss = None
    total_loss = 0
    total_loss_center = 0
    total_loss_seg = 0
    total_loss_offset = 0
    network = network.train()
    for i, (x, target) in enumerate(dataloader_train):
        #print("Train : len of data loader: ", len(dataloader_train))
        optimizer.zero_grad()
        loss_dict = network(x)
        losses = sum(loss_dict.values())

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses_reduced
        total_loss_center += loss_dict_reduced['loss_center']
        total_loss_seg += loss_dict_reduced['loss_sem_seg']
        total_loss_offset += loss_dict_reduced['loss_offset']

        scheduler.step()
        print("\rEpoch {} : Train Progress: {:>3.2f} %".format(epoch, ((i + 1) * config.no_gpus * 100) / len(dataloader_train)), end=' ')

    loss = {
        "loss_total": total_loss / (i+1),
        "loss_sem_seg": total_loss_seg / (i+1),
        "loss_center": total_loss_center / (i+1),
        "loss_offset": total_loss_offset / (i+1)
    }
    return loss

def eval_model(network, dataloader_val, epoch=None):
    loss = None
    total_loss = 0
    total_loss_center = 0
    total_loss_seg = 0
    total_loss_offset = 0
    network = network.train()
    for i, (x, target) in enumerate(dataloader_val):
        #print("Eval : len of data loader: ", len(dataloader_val))

        loss_dict = network(x)
        losses = sum(loss_dict.values())

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        total_loss += losses_reduced
        total_loss_center += loss_dict_reduced['loss_center']
        total_loss_seg += loss_dict_reduced['loss_sem_seg']
        total_loss_offset += loss_dict_reduced['loss_offset']

        print("\rEpoch {} : Val Progress: {:>3.2f} %".format(epoch, ((i + 1) * config.no_gpus * 100)  / len(dataloader_val)), end=' ')

    loss = {
        "loss_total": total_loss / (i+1),
        "loss_sem_seg": total_loss_seg / (i+1),
        "loss_center": total_loss_center / (i+1),
        "loss_offset": total_loss_offset / (i+1)
    }
    return loss

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


def training_routine(args, network, dataset_cfg):
    """Start OoD Training"""
    print("START OOD TRAINING: ", comm.get_local_rank())

    start_epoch = config.start_epoch
    epochs = config.training_epoch
    ckpt_path = config.ckpt_path


    dataset_train = CityscapesOOD(root=config.cityscapes_ood_path, split=config.split, cfg=dataset_cfg)
    dataset_val = CityscapesOOD(root=config.cityscapes_ood_path, split="val", cfg=dataset_cfg)
    start = time.time()

    # network = torch.nn.DataParallel(network).cuda()
    network = network.cuda()

    transform = Compose([RandomHorizontalFlip(), ToTensor(),
                         Normalize(dataset_train.mean, dataset_train.std)])

    optimizer = build_optimizer(dataset_cfg, network)
    scheduler = build_lr_scheduler(dataset_cfg, optimizer)

    print("Checkpoint file:", ckpt_path)
    if ckpt_path is not None:
        DetectionCheckpointer(network).resume_or_load(
            ckpt_path, resume=True
        )

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                            collate_fn=panoptic_deep_lab_collate)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=panoptic_deep_lab_collate)

    losses_total_train = []
    losses_total_val = []
    losses_seg = []
    losses_center = []
    losses_offset = []
    best_val_loss = 999
    best_epoch = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""

        network.train()
        train_loss = train_model(network, dataloader_train, optimizer, scheduler, epoch)
        torch.cuda.empty_cache()
        val_loss = eval_model(network, dataloader_val, epoch)
        if comm.is_main_process():
            print("\rEpoch {} with Train loss = {} and Val loss = {}".format(epoch, train_loss['loss_total'], val_loss['loss_total']))
        losses_total_train.append(train_loss["loss_total"])
        losses_total_val.append(val_loss["loss_total"])
        losses_seg.append(train_loss["loss_sem_seg"])
        losses_center.append(train_loss["loss_center"])
        losses_offset.append(train_loss["loss_offset"])

        torch.cuda.empty_cache()

        if comm.is_main_process() and val_loss["loss_total"] < best_val_loss:
            best_val_loss = val_loss["loss_total"]
            best_epoch = epoch

            """Save model state"""
            save_basename = config.model_name + "_best_model.pth"
            print('Saving checkpoint', os.path.join(config.weights_dir, save_basename))
            torch.save({
                'model': network.state_dict(),
                'epoch': epoch
            }, os.path.join(config.weights_dir, save_basename))

            '''network.eval()
            result = eval_metric_model(network)
            print("IoU: ", result["semantic_seg"]["sem_seg"]["IoU"], ", PQ: ",
                  result["panotic_seg"]["panoptic_seg"]["PQ"])'''
        torch.cuda.empty_cache()

        if comm.is_main_process():
            x_values = [i for i in range(start_epoch, epoch + 1)]
            fig = plt.figure("Total loss " + str(epoch))
            plt.plot(x_values, losses_total_train, label="Train loss")
            plt.plot(x_values, losses_total_val, label="Val loss")
            plt.title("Total loss")
            plt.legend()
            fig.savefig("./total_loss.png")

            fig = plt.figure("Semantic loss " + str(epoch))
            plt.plot(x_values, losses_seg)
            plt.title("Semantic loss")
            fig.savefig("./semantic_loss.png")

            fig = plt.figure("Center loss " + str(epoch))
            plt.plot(x_values, losses_center)
            plt.title("Center loss")
            fig.savefig("./center_loss.png")

            fig = plt.figure("Offset loss " + str(epoch))
            plt.plot(x_values, losses_offset)
            plt.title("Offset loss")
            fig.savefig("./offset_loss.png")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    if comm.is_main_process():
        print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        print("Best epoch: ", best_epoch)
        print(losses_total_train)
        print(losses_total_val)
        print(losses_seg)
        print(losses_center)
        print(losses_offset):q

def main(args):
    # load configuration from cfg files for detectron2
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config.detectron_config_file_path)
    cfg.freeze()
    default_setup(
        cfg, args
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    network = build_model(cfg)
    print("Model:\n{}".format(network))

    distributed = comm.get_world_size() > 1
    if distributed:
        network = DistributedDataParallel(
            network, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    """Perform training"""
    training_routine(args, network, cfg)


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
    print("Command Line Args:", args)
    launch(
        main,
        config.no_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    # main(vars(parser.parse_args()))
