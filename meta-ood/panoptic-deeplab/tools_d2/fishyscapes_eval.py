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

from config import config_evaluation_setup
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


class AnomalyDetector():
    def __init__(self, model=None, dataset=None, transform=None, model_name=None):
        self.network = model
        self.dataset = dataset
        self.transform = transform
        self.model_name = model_name

    def estimator_worker(self, image):

        input = self.preprocess_image(image)
        output = self.network(input)

        if isinstance(output, list):
            sem_out = F.softmax(output[0]['sem_seg'], 0)
            sem_out = sem_out.detach().cpu().numpy()
            # ent = np.max(sem_out, axis=0)
            ent = entropy(sem_out, axis=0) / np.log(self.dataset.num_eval_classes)
        else:
            sem_out = F.softmax(output[0])

            sem_out = sem_out.detach().cpu().numpy()

            ent = entropy(sem_out, axis=0) / np.log(self.dataset.num_eval_classes)
        '''import matplotlib.pyplot as plt
        plt.imshow(sem_out.argmax(axis=0))
        plt.savefig("/home/kumarasw/Thesis/segment.png")


        plt.imshow(torch.permute(image, (1, 2, 0)).numpy())
        plt.savefig("/home/kumarasw/Thesis/image.png")
        plt.imshow(ent)
        plt.savefig("/home/kumarasw/Thesis/mask.png")'''
        return torch.tensor(ent)

    def preprocess_image(self, x):

        if self.model_name == "DeepLabV3+_WideResNet38":

            x = np.array(Image.fromarray(np.array(x)).convert('RGB').resize((2048, 1024))).astype('uint8')
            x = Image.fromarray(x)
            x = standard_transforms.ToTensor()(x)
            x = standard_transforms.Normalize(self.dataset.mean, self.dataset.std)(x)
            x = x.cuda()
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
        else:
            x = np.array(Image.fromarray(np.array(x)).convert('RGB').resize((2048, 1024))).astype('uint8')
            x = Image.fromarray(x)
            x = standard_transforms.ToTensor()(x)
            x = standard_transforms.Normalize(self.dataset.mean, self.dataset.std)(x)
            x = [{"image": x, "height": x.size()[1], "width": x.size()[2]}]
        return x


def main(args):
    config = config_evaluation_setup(args.default_args)
    roots = config.roots
    params = config.params
    start_epoch = params.val_epoch
    dataset = config.dataset(root=config.roots.eval_dataset_root, model=config.roots.model_name)

    '''# load configuration from cfg files for detectron2
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
        network = load_network(model_name=roots.model_name, num_classes=19,
                               ckpt_path=roots.init_ckpt, train=False, cfg=cfg)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network = load_network(model_name=roots.model_name, num_classes=19,
                               ckpt_path=os.path.join(roots.weights_dir, basename), train=False, cfg=cfg)
        '''

    # This parameter is needed for panoptic segmentation data loading
    dataset_cfg = None

    # load configuration from cfg files for detectron2
    cfg = get_cfg()

    start_epoch = 0

    add_panoptic_deeplab_config(cfg)

    '''cfg.merge_from_file("/home/kumarasw/Thesis/meta-ood/src/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml")
    model_name = "Detectron_Panoptic_DeepLab"    
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"'''

    model_name = "Detectron_DeepLab"
    cfg.merge_from_file(
        "/home/kumarasw/Thesis/meta-ood/src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/Detectron_DeepLab_epoch_4_alpha_0.9.pth"

    '''model_name = "DeepLabV3+_WideResNet38"
    init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"
    #init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/cityscapes_best.pth"'''

    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=model_name, num_classes=19,
                               ckpt_path=init_ckpt, train=False, cfg=cfg)

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
    transform = Compose([ToTensor(), Normalize(config.dataset.mean, config.dataset.std)])
    detector = AnomalyDetector(network, config.dataset, transform, model_name)
    metrics = fs.evaluate(detector.estimator_worker, ds)

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

    '''launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )'''




