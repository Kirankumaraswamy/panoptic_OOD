import os
import sys

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from src.model.deepv3 import DeepWV3Plus
from src.model.DualGCNNet import DualSeg_res50
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import config
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.engine import DefaultTrainer



def load_network(model_name, num_classes, ckpt_path=None, train=False, cfg=None):
    network = None

    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)
    if model_name == "DeepLabV3+_WideResNet38":
        network = nn.DataParallel(DeepWV3Plus(num_classes))
        #network = DeepWV3Plus(num_classes)
    elif model_name == "DualGCNNet_res50":
        network = DualSeg_res50(num_classes)
    elif model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
        cfg = get_cfg()
        if model_name == "Detectron_DeepLab":
            add_deeplab_config(cfg)
            cfg.merge_from_file(config.Detectron_DeepLab_Config)
        elif model_name == "Detectron_Panoptic_DeepLab":
            add_panoptic_deeplab_config(cfg)
            cfg.merge_from_file(config.Detectron_PanopticDeepLab_Config)
        network = build_model(cfg)
    else:
        print("\nModel is not known")
        exit()

    if ckpt_path is not None:
        if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
            DetectionCheckpointer(network).resume_or_load(
                ckpt_path, resume=False
            )
        else:
            #network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
            network.load_state_dict(torch.load(ckpt_path)['model'], strict=False)
    
    #network = torch.nn.DataParallel(network).cuda()
    network = network.cuda()
    if train:
        print("... ok")
        return network.train()
    else:
        print("... ok")
        return network.eval()


def prediction(net, image):
    sem_out = None
    panoptic_out = None
    center_heat_map = None
    if not isinstance(image, list):
        image = image.cuda()
    net.eval()
    print(image.size())
    with torch.no_grad():
        out = net(image)
    if isinstance(out, tuple):
        sem_out = out[0]
    elif isinstance(out, list):
        if "sem_seg" in out[0].keys():
            sem_out = out[0]['sem_seg']
        if len(out) >= 2 and "panoptic_seg" in out[1].keys():
            panoptic_out = out[1]['panoptic_seg'][0]
            panoptic_out = panoptic_out.data.cpu()
            panoptic_out = np.squeeze(panoptic_out.numpy())
            center_heat_map = out[1]['center_heat_map']
            center_heat_map = center_heat_map.data.cpu()
            center_heat_map = np.squeeze(center_heat_map.numpy())
    else:
        sem_out = np.squeeze(out)
    sem_out = sem_out.data.cpu()
    sem_out = F.softmax(sem_out, 0)
    return sem_out.numpy(), panoptic_out, center_heat_map


class inference(object):

    def __init__(self, params, roots, loader, num_classes=None, init_net=True):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.model_name = roots.model_name
        self.batch = 0
        self.batch_max = int(len(loader) / self.batch_size) + (len(loader) % self.batch_size > 0)
        self.loader = loader
        self.batchloader = iter(DataLoader(loader, batch_size=self.batch_size, shuffle=False))
        self.probs_root = os.path.join(roots.io_root, "probs")

        if self.epoch == 0:
            pattern = "baseline"
            ckpt_path = roots.init_ckpt
            self.probs_load_dir = os.path.join(self.probs_root, pattern)
        else:
            pattern = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha)
            basename = self.model_name + "_" + pattern + ".pth"
            self.probs_load_dir = os.path.join(self.probs_root, pattern)
            ckpt_path = os.path.join(roots.weights_dir, basename)
        if init_net and num_classes is not None:
            self.net = load_network(self.model_name, num_classes, ckpt_path)

    def probs_gt_load(self, i, load_dir=None):
        if load_dir is None:
            load_dir = self.probs_load_dir
        try:
            filename = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
            f_probs = h5py.File(filename, "r")
            probs = np.asarray(f_probs['probabilities'])
            gt_train = np.asarray(f_probs['gt_train_ids'])
            gt_label = np.asarray(f_probs['gt_label_ids'])
            probs = np.squeeze(probs)
            gt_train = np.squeeze(gt_train)
            gt_label = np.squeeze(gt_label)
            im_path = f_probs['image_path'][0].decode("utf8")
        except OSError:
            print("No probs file for image %d, therefore run inference..." % i)
            probs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        return probs, gt_train, gt_label, im_path

    def probs_gt_save(self, i, save_dir=None):
        if save_dir is None:
            save_dir = self.probs_load_dir
        if not os.path.exists(save_dir):
            print("Create directory:", save_dir)
            os.makedirs(save_dir)
        probs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        file_name = os.path.join(save_dir, "probs" + str(i) + ".hdf5")
        f = h5py.File(file_name, "w")
        f.create_dataset("probabilities", data=probs)
        f.create_dataset("gt_train_ids", data=gt_train)
        f.create_dataset("gt_label_ids", data=gt_label)
        f.create_dataset("image_path", data=[im_path.encode('utf8')])
        print("file stored:", file_name)
        f.close()

    def probs_gt_load_batch(self):
        assert self.batch_size > 1, "Please use batch size > 1 or use function 'probs_gt_load()' instead, bye bye..."
        x, y, z, im_paths = next(self.batchloader)
        probs = prediction(self.net, x)
        gt_train = y.numpy()
        gt_label = z.numpy()
        self.batch += 1
        print("\rBatch %d/%d processed" % (self.batch, self.batch_max))
        sys.stdout.flush()
        return probs, gt_train, gt_label, im_paths

    def prob_gt_calc(self, i):
        x, y = self.loader[i]
        if not isinstance(x, list):
            x = x.unsqueeze_(0)

        '''import matplotlib.pyplot as plt
        plt.imshow(x[0]["image"].permute(1, 2, 0).numpy())
        plt.savefig("/home/kumarasw/Thesis/driving_uncertainty/image.png")'''
        sem_seg, panoptic_seg, center_heat_map = prediction(self.net, x)
        probs = sem_seg
        gt_train = y.numpy()
        '''import matplotlib.pyplot as plt
        plt.imshow(x[0]["image"].permute(1, 2, 0).numpy())
        #plt.imshow(x.squeeze().permute(1, 2, 0).numpy())
        plt.show()
        out = probs.argmax(axis=0)
        plt.imshow(out)
        plt.show()'''

        try:
            gt_label = np.array(Image.open(self.loader.annotations[i]).convert('L'))
        except AttributeError:
            gt_label = np.zeros(gt_train.shape)
        im_path = self.loader.images[i]
        return probs, gt_train, gt_label, im_path


def probs_gt_load(i, load_dir):
    try:
        filepath = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
        f_probs = h5py.File(filepath, "r")
        probs = np.asarray(f_probs['probabilities'])
        gt_train = np.asarray(f_probs['gt_train_ids'])
        gt_label = np.asarray(f_probs['gt_label_ids'])
        probs = np.squeeze(probs)
        gt_train = np.squeeze(gt_train)
        gt_label = np.squeeze(gt_label)
        im_path = f_probs['image_path'][0].decode("utf8")
    except OSError:
        probs, gt_train, gt_label, im_path = None, None, None, None
        print("No probs file, see src.model_utils")
        exit()
    return probs, gt_train, gt_label, im_path

