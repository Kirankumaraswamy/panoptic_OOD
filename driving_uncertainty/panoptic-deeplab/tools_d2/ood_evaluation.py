import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import yaml
import random
from options.config_class import Config
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
import tensorflow as tf
import seaborn as sns
import sys
import _init_paths
import d2
sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_segmentation'))
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_synthesis'))
from image_synthesis.models.pix2pix_model import Pix2PixModel
from image_dissimilarity.models.dissimilarity_model import DissimNetPrior, DissimNet
from image_dissimilarity.models.vgg_features import VGG19_difference
from image_dissimilarity.data.cityscapes_dataset import one_hot_encoding
from panoptic_evaluation.evaluation import data_load, data_evaluate
from panoptic_evaluation.cityscapes_ood import CityscapesOOD
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
from cityscapesscripts.helpers.labels import trainId2label
import matplotlib.pyplot as plt
import ood_config


class AnomalyDetector():
    def __init__(self, ours=True, seed=0, fishyscapes_wrapper=True, ood_threshold=None):

        self.set_seeds(seed)

        # Common options for all models
        TestOptions = Config()
        self.opt = TestOptions
        torch.cuda.empty_cache()
        self.get_segmentation()
        self.get_synthesis()
        self.get_dissimilarity(ours)
        self.get_transformations()
        self.fishyscapes_wrapper = fishyscapes_wrapper
        self.ood_threshold = ood_threshold

    def estimator_image(self, image):
        image_og_h = image.size[1]
        image_og_w = image.size[0]
        img = image.resize((2048, 1024))
        img_tensor = self.img_transform(img)

        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())

        seg_softmax_out = F.softmax(seg_outs, dim=1)
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map

        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity

        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity

        del seg_softmax_out
        torch.cuda.empty_cache()

        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))

        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}

        generated = self.syn_net(syn_input, mode='inference')

        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))

        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))

        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))

        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()

        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()

        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]

        # Resize outputs to original input image size
        diss_pred = Image.fromarray(diss_pred.squeeze() * 255).resize((image_og_w, image_og_h))
        seg_img = semantic.resize((image_og_w, image_og_h))
        entropy = entropy_img.resize((image_og_w, image_og_h))
        perceptual_diff = perceptual_diff.resize((image_og_w, image_og_h))
        distance = entropy.resize((image_og_w, image_og_h))
        synthesis = synthesis_final_img.resize((image_og_w, image_og_h))

        out = {'anomaly_map': diss_pred, 'segmentation': seg_img, 'synthesis': synthesis,
               'softmax_entropy': entropy, 'perceptual_diff': perceptual_diff, 'softmax_distance': distance}

        return out

    # Loop around all figures
    def estimator_worker(self, image):
        image=image[0]["image"]
        img = Image.fromarray(np.array(image))

        img_tensor = self.img_transform(img)
        image_og_h = img_tensor.shape[1]
        image_og_w = img_tensor.shape[2]

        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())

        seg_softmax_out = F.softmax(seg_outs, dim=1)
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map

        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity

        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity

        del seg_softmax_out
        torch.cuda.empty_cache()

        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))

        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        generated = self.syn_net(syn_input, mode='inference')

        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))

        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))

        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))

        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()

        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()

        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = np.array(Image.fromarray(diss_pred.squeeze()).resize((image_og_w, image_og_h)))

        seg_final[np.where(diss_pred > 0.1)] = 19

        out = [{'anomaly_score': torch.tensor(diss_pred), 'sem_seg': torch.tensor(seg_final)}]


        return out

    def detectron_estimator_worker(self, input):
        image = input[0]["image"]
        img = Image.fromarray(np.array(image.permute((1,2,0))))
        #img_tensor = self.img_transform(img)
        #data = [{"image": img_tensor, "height": img_tensor.size()[1], "width": img_tensor.size()[2]}]
        self.img = img
        self.seg_net.evaluate_ood = ood_config.evaluate_ood
        self.seg_net.synboost = self
        self.seg_net.ood_threshold=threshold
        self.seg_net.read_instance_path = ood_config.read_instance_path
        self.seg_net.performance_with_ood = ood_config.performance_with_ood
        output = self.seg_net(input)
        output[0]["sem_seg"] = output[0]["sem_seg"].cpu()
        if ood_config.evaluate_ood:
            output[0]["anomaly_score"] = output[0]["anomaly_score"].cpu()

        '''plt.imshow(torch.squeeze(output[0]["anomaly_score"].detach().cpu()).numpy())
        plt.show()'''
        
        if ood_config.save_results:
            self.display_results(input, output)
        return output

    def synboost_uncertainity(self, seg_softmax_out):

        img = self.img
        image_og_h = seg_softmax_out.squeeze().shape[1]
        image_og_w = seg_softmax_out.squeeze().shape[2]

        seg_final = np.argmax(seg_softmax_out.detach().cpu().numpy(), axis=0)
        # segmentation map
        seg_softmax_out = torch.unsqueeze(seg_softmax_out, dim=0)

        # get entropy
        # added small noise to overcome log 0
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out+0.00001), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity

        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity

        del seg_softmax_out
        torch.cuda.empty_cache()

        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))

        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        generated = self.syn_net(syn_input, mode='inference')

        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))

        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))

        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))

        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()

        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()

        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = np.array(Image.fromarray(diss_pred.squeeze()).resize((image_og_w, image_og_h)))

        seg_final[np.where(diss_pred > self.ood_threshold)] = 19

        out = {'anomaly_score': torch.tensor(diss_pred), 'sem_seg': torch.tensor(seg_final)}
        

        return out

    def display_results(self, image, output):

        image_save_dir = os.path.join(".", "ood_results")
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        image_save_path = os.path.join(image_save_dir, image[0]["image_id"] + ".png")

        fig = plt.figure(figsize=(20, 14))
        rows = 2
        columns = 3
        images = []
        img1 = np.array(image[0]["real_image"])

        img2 = output[0]["sem_seg"].detach().cpu().squeeze().numpy()

        img3 = output[0]["anomaly_score"].detach().cpu().squeeze().numpy()

        pan_img = output[0]["panoptic_seg"][0].detach().cpu().squeeze().numpy()

        segment_ids = np.unique(pan_img)
        pan_format = np.zeros(img1.shape, dtype="uint8")
        for segmentId in segment_ids:
            if segmentId > 1000:
                semanticId = segmentId // 1000
                labelInfo = trainId2label[semanticId]
                if labelInfo.hasInstances:
                    mask = np.where(pan_img == segmentId)
                    color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                    pan_format[mask] = color
        img4 = pan_format

        img5 = output[0]["centre_score"].detach().cpu().squeeze().numpy()

        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)

        for i in range(5):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

        fig.tight_layout()
        plt.savefig(image_save_path)



    def set_seeds(self, seed=0):
        # set seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def get_segmentation(self):
        assert_and_infer_cfg(self.opt, train_mode=False)
        self.opt.dataset_cls = cityscapes

        model_name = ood_config.model_name
        ckpt_path = ood_config.init_ckpt
        config_file = ood_config.config_file

        train = False

        print("Checkpoint file:", ckpt_path)
        print("Load model:", model_name, end="", flush=True)

        if model_name == "Detectron_DeepLab" or model_name == "Detectron_Panoptic_DeepLab":
            cfg = get_cfg()
            if model_name == "Detectron_DeepLab":
                add_deeplab_config(cfg)
                cfg.merge_from_file(ood_config.config_file)
            elif model_name == "Detectron_Panoptic_DeepLab":
                add_panoptic_deeplab_config(cfg)
                cfg.merge_from_file(ood_config.config_file)
            network = build_model(cfg)
            #network = torch.nn.DataParallel(network).cuda()
            DetectionCheckpointer(network).resume_or_load(
                ckpt_path, resume=False
            )
            self.seg_net = network.cuda()
            if train:
                print("... ok")
                self.seg_net.train()
            else:
                print("... ok")
                self.seg_net.eval()


        else:
            # Get Segmentation Net
            net = network.get_net(self.opt, criterion=None)
            net = torch.nn.DataParallel(net).cuda()
            print('Segmentation Net Built.')
            snapshot = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.snapshot)
            self.seg_net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot,
                                               restore_optimizer_bool=False)
            self.seg_net.eval()
            print('Segmentation Net Restored.')

    def get_synthesis(self):
        # Get Synthesis Net
        print('Synthesis Net Built.')
        self.opt.checkpoints_dir = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.checkpoints_dir)
        self.syn_net = Pix2PixModel(self.opt)
        self.syn_net.eval()
        print('Synthesis Net Restored')

    def get_dissimilarity(self, ours):
        # Get Dissimilarity Net
        if ours:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                       'image_dissimilarity/configs/test/ours_configuration.yaml')
        else:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                       'image_dissimilarity/configs/test/baseline_configuration.yaml')

        with open(config_diss, 'r') as stream:
            config_diss = yaml.load(stream, Loader=yaml.FullLoader)

        self.prior = config_diss['model']['prior']
        self.ensemble = config_diss['ensemble']

        if self.prior:
            self.diss_model = DissimNetPrior(**config_diss['model']).cuda()
        else:
            self.diss_model = DissimNet(**config_diss['model']).cuda()

        print('Dissimilarity Net Built.')
        save_folder = os.path.join(os.getcwd(), os.path.dirname(__file__), config_diss['save_folder'])
        model_path = os.path.join(save_folder,
                                  '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
        model_weights = torch.load(model_path)
        self.diss_model.load_state_dict(model_weights)
        self.diss_model.eval()
        print('Dissimilarity Net Restored')

    def get_transformations(self):
        # Transform images to Tensor based on ImageNet Mean and STD
        #mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

        # synthesis necessary pre-process
        self.transform_semantic = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
        self.transform_image_syn = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.BICUBIC), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])

        # dissimilarity pre-process
        self.vgg_diff = VGG19_difference().cuda()
        self.base_transforms_diss = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
        self.norm_transform_diss = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
        self.to_pil = ToPILImage()


def visualize_tfdataset(tfdataset, num_samples):
    """Visualizes `num_samples` from the `tfdataset`."""

    fig, axs = plt.subplots(num_samples, 2, figsize=(7, 2 * num_samples))
    for i, blob in enumerate(tfdataset.take(num_samples)):
        image = blob['image_left'].numpy()
        mask = blob['mask'].numpy()
        axs[i][0].imshow(image.astype('int'))
        axs[i][0].axis("off")
        axs[i][0].set(title="Image")
        # map 255 to 2 such that difference between labels is better visible
        mask[mask == 255] = 2
        axs[i][1].imshow(mask[..., 0])
        axs[i][1].axis("off")
        axs[i][1].set(title="Mask")
    fig.show()

def panoptic_deep_lab_collate(batch):
    data = [item for item in batch]
    return data


if __name__ == '__main__':
    transform = None

    #thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    thresholds = [ood_config.ood_threshold]
    specificity = []
    sensitivity = []
    gmean = []
    detector = AnomalyDetector(True, ood_threshold=None)
    ds = data_load(root=ood_config.ood_dataset_path, split=ood_config.ood_split,
                   transform=transform)
    pq_in = []
    pq_out = []
    pod_q = []
    for threshold in thresholds:
        if ood_config.evaluate_ood:
            print("====================================================")
            print("              Threshold: ", threshold)
            print("====================================================")
        
        detector.ood_threshold=threshold
        # This evaluation works only with batch size 1
        result = data_evaluate(estimator=detector.detectron_estimator_worker, evaluation_dataset=ds, batch_size=1,
                               collate_fn=panoptic_deep_lab_collate, evaluate_ood=ood_config.evaluate_ood, semantic_only=ood_config.semantic_only, evaluate_anomoly = ood_config.evaluate_anomoly)
        print("====================================================")
        print(result)
        specificity.append(result['semantic_seg']['sem_seg']['uSpecificity'])
        sensitivity.append(result['semantic_seg']['sem_seg']['uSensitivity'])
        gmean.append(result['semantic_seg']['sem_seg']['uGmean'])
        
        pq_in.append(result['panotic_seg']['panoptic_seg']['PQ-in'])
        pq_out.append(result['panotic_seg']['panoptic_seg']['PQ-out'])
        pod_q.append(result['panotic_seg']['panoptic_seg']['POD-Q'])

    if ood_config.evaluate_ood:
        if len(thresholds) > 1:
            default_x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            fig = plt.figure()
            plt.plot(thresholds, specificity,  label="uSpecificity")
            plt.plot(thresholds, sensitivity,  label="uSensitivity")
            plt.plot(thresholds, gmean, label="G-Mean")
            plt.xlabel("Threshold")
            plt.ylabel("Performance scaled to 1")
            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()
            plt.show()
            fig.savefig("./synbbost_sensitivity_vs_specificity_val.png",dpi=200)

            fig = plt.figure()
            plt.plot(thresholds, pq_in, label="PQ-in")
            plt.plot(thresholds, pq_out, label="PQ-out")
            plt.plot(thresholds, pod_q, label="POD-Q")
            plt.xlabel("Threshold")
            plt.ylabel("Performance in %")

            plt.xticks(default_x_ticks, default_x_ticks, ha='center')
            plt.legend()

            fig.savefig("./synboost_podq_threshold_val.png", dpi=200)

        print("Thresholds: ", thresholds)
        print("Gmean: ", gmean)
        print('Usensitivity: ', sensitivity)
        print("Uspecivicity: ", specificity)

        print("============================================")
        print("PQ-in", pq_in)
        print("PQ-out", pq_out)
        print("POD-Q", pod_q)
