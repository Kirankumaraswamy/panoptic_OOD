import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import entropy
import tensorflow_datasets as tfds
import torchvision.transforms as standard_transforms
from PIL import Image
import tensorflow as tf

from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
import ood_config
from panoptic_evaluation.cityscapes_ood import CityscapesOOD
from torchvision.transforms import Compose, ToTensor, Normalize

import _init_paths
import d2

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
    def __init__(self, model=None, transform=None):
        self.network = model
        self.transform = transform

    def estimator_worker(self, image):
        image = self.preprocess_image(image)
        with torch.no_grad():
            output = self.network(image)
        anomoaly_score = output[0]['anomaly_score']
        '''import matplotlib.pyplot as plt
        plt.imshow(anomoaly_score.detach().cpu().numpy().squeeze())
        plt.show()'''

        return anomoaly_score.detach().cpu()

    def preprocess_image(self, x):
        x = np.array(Image.fromarray(np.array(x)).convert('RGB').resize((2048, 1024))).astype('uint8')
        x = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
        x = [{"image": x, "height": x.size()[1], "width": x.size()[2]}]
        return x


def main(args):

    net = get_net()

    net.evaluate_ood = True

    #transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)])
    transform = None
    net.ood_threshold = ood_config.ood_threshold
    detector = AnomalyDetector(net)


    import bdlb
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)

    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    fs.download_and_prepare('LostAndFound')
    # fs.download_and_prepare('Static')
    # automatically downloads the dataset

    ds = tfds.load('fishyscapes/LostAndFound', split='validation')
    # ds = tfds.load('Static')

    # data = fs.get_dataset('Static')
    #transform = Compose([ToTensor(), Normalize(CityscapesOOD.mean, CityscapesOOD.std)])
    metrics = fs.evaluate(detector.estimator_worker, ds)
    print(metrics)
    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% FPR@95TPR'.format(100 * metrics['FPR@95%TPR']))
    print('My method achieved {:.2f}% auroc'.format(100 * metrics['auroc']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-val", "--VALSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--val_epoch", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-pixel", "--pixel_eval", action='store_true')
    parser.add_argument("-segment", "--segment_eval", action='store_true')

    # use detectron2 distributed args
    args = default_argument_parser().parse_args()
    # args from current file
    args.default_args = vars(parser.parse_args())

    main(args)
