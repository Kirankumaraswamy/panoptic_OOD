#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

#import config as config    
import _init_paths
import d2
import glob

def _print_panoptic_results(pq_res, f):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    f.write("Panoptic Evaluation Results:\n" + table)
    
def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        if min(cfg.INPUT.CROP.SIZE) > min(cfg.INPUT.MIN_SIZE_TRAIN):
            augs.append(T.MyOpTransform(cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        #if evaluator_type == "cityscapes_panoptic_seg":
        #    evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        #    evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def logging(res, checkpoint, f, print_flag=False):
    if print_flag:
        log_text = "----------------------------------------\n Best Checkpoint Results "
        log_text += "Evaluating {}:".format(checkpoint)
    else:
        log_text = "Evaluating {}:".format(checkpoint)
    for key in res.keys():
        log_text += "{} : {} ".format(key, res[key])
    log_text += "\n"
    f.write(log_text)
    if print_flag:
        print (log_text)
    

def main(args):
    cfg = setup(args)
    model = Trainer.build_model(cfg)

    PQ_best = 0
    best_checkpoint = None
    best_res = None

    #logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, 'log_find_best.txt'))
    if comm.is_main_process():
        f = open(os.path.join(cfg.OUTPUT_DIR, 'log_find_best.txt'), 'w')
    for checkpoint in sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, 'model_*.pth'))):
        if '.pth' not in checkpoint:
            continue
        checkpoint_name = checkpoint.split('/')[-1]
        if comm.is_main_process():
            print ('---------- Evaluating: ' + checkpoint_name)
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            checkpoint, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process() and 'panoptic_seg' in res.keys():
            PQ = res['panoptic_seg']['PQ']
            logging(res['panoptic_seg'], checkpoint_name, f)

            if PQ > PQ_best:
                PQ_best = PQ
                best_checkpoint = checkpoint_name
                best_res = res['panoptic_seg']

    if comm.is_main_process():
        logging(best_res, best_checkpoint, f, True)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.dist_url = 'tcp://127.0.0.1:64487'
    print("Command Line Args:", args)
    #os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
