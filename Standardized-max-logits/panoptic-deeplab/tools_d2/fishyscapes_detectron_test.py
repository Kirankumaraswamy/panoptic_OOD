import bdlb
import argparse
import os
from PIL import Image
import network
import numpy as np

from network import mynn
import argparse
import logging
import os
import torch
import time
import numpy as np
import tensorflow as tf

from config import cfg, assert_and_infer_cfg
import network
import optimizer
from tqdm import tqdm

from PIL import Image
import torchvision.transforms as standard_transforms
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
from detectron2.engine import DefaultTrainer

import math
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from scipy import ndimage as ndi
from kornia.morphology import dilation, erosion
import _init_paths
import d2


dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, 'pretrained/r101_os8_base_cty.pth')

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='possible datasets for statistics; cityscapes')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')

parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')

parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

parser.add_argument('--ood_dataset_path', type=str,
                    default='/export/kiran/fishy_scapes_lost_and_found/test',
                    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='standardized_max_logit',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

args.world_size = 1

print(f'World Size: {args.world_size}')
if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)


# NOTE(shjung13): These are for obtaining non-boundary masks
# We calculate the boundary mask by subtracting the eroded prediction map from the dilated one
# These are filters for erosion and dilation (L1)
selem = torch.ones((3, 3)).cuda()
selem_dilation = torch.FloatTensor(ndi.generate_binary_structure(2, 1)).cuda()

print(f'selem:\n\n{selem}')
print(f'selem_dilation:\n\n{selem_dilation}')

# NOTE(shjung13): Dilation filters to expand the boundary maps (L1)
d_k1 = torch.zeros((1, 1, 2 * 1 + 1, 2 * 1 + 1)).cuda()
d_k2 = torch.zeros((1, 1, 2 * 2 + 1, 2 * 2 + 1)).cuda()
d_k3 = torch.zeros((1, 1, 2 * 3 + 1, 2 * 3 + 1)).cuda()
d_k4 = torch.zeros((1, 1, 2 * 4 + 1, 2 * 4 + 1)).cuda()
d_k5 = torch.zeros((1, 1, 2 * 5 + 1, 2 * 5 + 1)).cuda()
d_k6 = torch.zeros((1, 1, 2 * 6 + 1, 2 * 6 + 1)).cuda()
d_k7 = torch.zeros((1, 1, 2 * 7 + 1, 2 * 7 + 1)).cuda()
d_k8 = torch.zeros((1, 1, 2 * 8 + 1, 2 * 8 + 1)).cuda()
d_k9 = torch.zeros((1, 1, 2 * 9 + 1, 2 * 9 + 1)).cuda()

d_ks = {1: d_k1, 2: d_k2, 3: d_k3, 4: d_k4, 5: d_k5, 6: d_k6, 7: d_k7, 8: d_k8, 9: d_k9}

for k, v in d_ks.items():
    v[:,:,k,k] = 1
    for i in range(k):
        v = dilation(v, selem_dilation)
    d_ks[k] = v.squeeze(0).squeeze(0)

    print(f'dilation kernel at {k}:\n\n{d_ks[k]}')

# ckpt_path = "/home/kumarasw/Thesis/Standardized-max-logits/pretrained/deeplab_model_final_a8a355.pkl"
ckpt_path = "/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
model_name = "Detectron_Panoptic_DeepLab"
train = False
Detectron_PanopticDeepLab_Config = "/home/kumarasw/Thesis/Standardized-max-logits/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
Detectron_DeepLab_Config = "/home/kumarasw/Thesis/Standardized-max-logits/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"

def get_net():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)

    if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
        cfg = get_cfg()
        if model_name == "Detectron_DeepLab":
            add_deeplab_config(cfg)
            cfg.merge_from_file(Detectron_DeepLab_Config)
        elif model_name == "Detectron_Panoptic_DeepLab":
            add_panoptic_deeplab_config(cfg)
            cfg.merge_from_file(Detectron_PanopticDeepLab_Config)
        network = build_model(cfg)
        #network = torch.nn.DataParallel(network).cuda()
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
    seg_net = network.cuda()
    if train:
        print("... ok")
        seg_net.train()
    else:
        print("... ok")
        seg_net.eval()

    return seg_net

def find_boundaries(label):
    """
    Calculate boundary mask by getting diff of dilated and eroded prediction maps
    """
    assert len(label.shape) == 4
    boundaries = (dilation(label.float(), selem_dilation) != erosion(label.float(), selem)).float()
    ### save_image(boundaries, f'boundaries_{boundaries.float().mean():.2f}.png', normalize=True)

    return boundaries

def expand_boundaries(boundaries, r=0):
    """
    Expand boundary maps with the rate of r
    """
    if r == 0:
        return boundaries
    expanded_boundaries = dilation(boundaries, d_ks[r])
    ### save_image(expanded_boundaries, f'expanded_boundaries_{r}_{boundaries.float().mean():.2f}.png', normalize=True)
    return expanded_boundaries



class BoundarySuppressionWithSmoothing(nn.Module):
    """
    Apply boundary suppression and dilated smoothing
    """
    def __init__(self, boundary_suppression=True, boundary_width=4, boundary_iteration=4,
                 dilated_smoothing=True, kernel_size=7, dilation=6):
        super(BoundarySuppressionWithSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.boundary_suppression = boundary_suppression
        self.boundary_width = boundary_width
        self.boundary_iteration = boundary_iteration

        sigma = 1.0
        size = 7
        gaussian_kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
        gaussian_kernel /= np.sum(gaussian_kernel)
        gaussian_kernel = torch.Tensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        self.dilated_smoothing = dilated_smoothing

        self.first_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
        self.first_conv.weight = torch.nn.Parameter(torch.ones_like((self.first_conv.weight)))

        self.second_conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, dilation=self.dilation, bias=False)
        self.second_conv.weight = torch.nn.Parameter(gaussian_kernel)


    def forward(self, x, prediction=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_size = x.size()
        # B x 1 x H x W
        assert len(x.shape) == 4
        out = x
        if self.boundary_suppression:
            # obtain the boundary map of width 2 by default
            # this can be calculated by the difference of dilation and erosion
            boundaries = find_boundaries(prediction.unsqueeze(1))
            expanded_boundaries = None
            if self.boundary_iteration != 0:
                assert self.boundary_width % self.boundary_iteration == 0
                diff = self.boundary_width // self.boundary_iteration
            for iteration in range(self.boundary_iteration):
                if len(out.shape) != 4:
                    out = out.unsqueeze(1)
                prev_out = out
                # if it is the last iteration or boundary width is zero
                if self.boundary_width == 0 or iteration == self.boundary_iteration - 1:
                    expansion_width = 0
                # reduce the expansion width for each iteration
                else:
                    expansion_width = self.boundary_width - diff * iteration - 1
                # expand the boundary obtained from the prediction (width of 2) by expansion rate
                expanded_boundaries = expand_boundaries(boundaries, r=expansion_width)
                # invert it so that we can obtain non-boundary mask
                non_boundary_mask = 1. * (expanded_boundaries == 0)

                f_size = 1
                num_pad = f_size

                # making boundary regions to 0
                x_masked = out * non_boundary_mask
                x_padded = nn.ReplicationPad2d(num_pad)(x_masked)

                non_boundary_mask_padded = nn.ReplicationPad2d(num_pad)(non_boundary_mask)

                # sum up the values in the receptive field
                y = self.first_conv(x_padded)
                # count non-boundary elements in the receptive field
                num_calced_elements = self.first_conv(non_boundary_mask_padded)
                num_calced_elements = num_calced_elements.long()

                # take an average by dividing y by count
                # if there is no non-boundary element in the receptive field,
                # keep the original value
                avg_y = torch.where((num_calced_elements == 0), prev_out, y / num_calced_elements)
                out = avg_y

                # update boundaries only
                out = torch.where((non_boundary_mask == 0), out, prev_out)
                del expanded_boundaries, non_boundary_mask

            # second stage; apply dilated smoothing
            if self.dilated_smoothing == True:
                out = nn.ReplicationPad2d(self.dilation * 3)(out)
                out = self.second_conv(out)

            return out.squeeze(1)
        else:
            if self.dilated_smoothing == True:
                out = nn.ReplicationPad2d(self.dilation * 3)(out)
                out = self.second_conv(out)
            else:
                out = x

        return out.squeeze(1)

class AnomalyDetector:

    def __init__(self, model=None, mean_std=None, num_classes=19, class_mean=None, class_var=None):
        self.model = model
        self.mean_std = mean_std
        self.class_mean = class_mean
        self.class_var = class_var
        self.num_classes = num_classes

        self.multi_scale = BoundarySuppressionWithSmoothing(
                    boundary_suppression=args.enable_boundary_suppression,
                    boundary_width=args.boundary_width,
                    boundary_iteration=args.boundary_iteration,
                    dilated_smoothing=args.enable_dilated_smoothing,
                    kernel_size=args.smoothing_kernel_size,
                    dilation=args.smoothing_kernel_dilation)
        self.multi_scale = self.multi_scale.cuda()
    
    def estimator_worker(self, image):
        image = np.array(Image.fromarray(np.array(image)).convert('RGB').resize((2048, 1024))).astype('uint8')
        with torch.no_grad():
            image = self.preprocess_image(image, self.mean_std)
            image = torch.squeeze(image)
            input = [{"image": image, "height": image.size()[1], "width": image.size()[2]}]
            main_out = self.model(input)

            main_out = torch.unsqueeze(main_out[0]['sem_seg'], dim=0)


            if self.class_mean is None or self.class_var is None:
                raise Exception("Class mean and var are not set!")
            anomaly_score, prediction = main_out.detach().max(1)
            for c in range(self.num_classes):
                anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - self.class_mean[c]) / np.sqrt(self.class_var[c]),
                                            anomaly_score)
            
            '''with torch.no_grad():
                anomaly_score = self.multi_scale(anomaly_score, prediction)'''

        del main_out
        anomaly_score = anomaly_score * -1

        return anomaly_score.cpu()

    def preprocess_image(self, x, mean_std):
        x = Image.fromarray(x)
        x = standard_transforms.ToTensor()(x)
        x = standard_transforms.Normalize(*mean_std)(x)

        x = x.cuda()

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x



if __name__ == '__main__':
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    net = get_net()

    class_mean = np.load(f'./stats/{args.dataset}_{model_name}_mean.npy', allow_pickle=True)
    class_var = np.load(f'./stats/{args.dataset}_{model_name}.npy', allow_pickle=True)
    
    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    fs.download_and_prepare('LostAndFound')
    #fs.download_and_prepare('Static')
    # automatically downloads the dataset

    ds = tfds.load('fishyscapes/LostAndFound', split='validation')

    detector = AnomalyDetector(net, mean_std, class_mean=class_mean.item(), class_var=class_var.item())
    metrics = fs.evaluate(detector.estimator_worker, ds)

    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% FPR@95TPR'.format(100 * metrics['FPR@95%TPR']))
    print('My method achieved {:.2f}% auroc'.format(100 * metrics['auroc']))