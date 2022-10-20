import os
from src.dataset.cityscapes_coco_mixed import CityscapesCocoMix
from src.dataset.lost_and_found import LostAndFound
from src.dataset.fishyscapes import Fishyscapes

TRAINSETS   = ["Cityscapes+COCO"]
VALSETS     = ["LostAndFound", "Fishyscapes"]
MODELS      = ["DeepLabV3+_WideResNet38", "DualGCNNet_res50", "Detectron_DeepLab", "Detectron_Panoptic_DeepLab"]

TRAINSET    = TRAINSETS[0]
VALSET      = VALSETS[1]
MODEL       = MODELS[3]
#IO          = "/home/chan/io/ood_detection/"
IO = "/home/mohan/kiran/ponoptic_OOD/meta-ood/results"
Detectron_PanopticDeepLab_Config = "/home/mohan/kiran/ponoptic_OOD/meta-ood/src/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
Detectron_DeepLab_Config = "/home/mohan/kiran/ponoptic_OOD/meta-ood/src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"

class cs_coco_roots:
    """
    OoD training roots for Cityscapes + COCO mix
    """
    model_name  = MODEL
    #init_ckpt = "/home/kumarasw/original-meta/meta-ood/weights/cityscapes_best.pth"
    init_ckpt   = "/home/mohan/kiran/ponoptic_OOD/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
    #init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/deeplab_model_final_a8a355.pkl"
    cs_root     = "/home/mohan/kiran/dataset/cityscapes/"
    coco_root   = "/home/mohan/kiran/dataset/coco/2017/"
    io_root     = IO + "meta_ood_" + model_name
    weights_dir = os.path.join(io_root, "weights/")


class laf_roots:
    """
    LostAndFound config class
    """
    model_name = MODEL
    #init_ckpt = os.path.join("/home/chan/io/cityscapes/weights/", model_name + ".pth")
    init_ckpt = os.path.join("/home/kiran/kiran/Thesis/OOD/code/meta-ood", model_name + ".pth")
    eval_dataset_root = "/home/datasets/lost_and_found/"
    #eval_dataset_root = "/home/kiran/kiran/Thesis/code/datasets/fishyscapes_lostandfound"
    eval_sub_dir = "laf_eval"
    io_root = os.path.join(IO + "meta_ood_" + model_name, eval_sub_dir)
    #weights_dir = os.path.join(io_root, "..", "weights/")
    weights_dir = os.path.join("/home/kiran/kiran/Thesis/OOD/code/meta-ood")


class fs_roots:
    """
    Fishyscapes config class
    """
    model_name = MODEL
    #init_ckpt = os.path.join("/home/kumarasw/meta-ood/weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth")
    #init_ckpt = "/home/kumarasw/original-meta/meta-ood/weights/cityscapes_best.pth"
    #init_ckpt = "/home/kumarasw/Thesis/meta-ood/weights/deeplab_model_final_a8a355.pkl"
    init_ckpt ="/home/kumarasw/Thesis/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
    # eval_dataset_root = "/home/datasets/fishyscapes/"
    eval_dataset_root = "/export/kiran/fishy_scapes_lost_and_found/"
    eval_sub_dir = "fs_eval"
    io_root = os.path.join("/home/kumarasw/Thesis/meta-ood/")
    weights_dir = os.path.join(io_root, "weights/")


class params:
    """
    Set pipeline parameters
    """
    training_starting_epoch = 0
    num_training_epochs     = 4
    pareto_alpha            = 0.9
    ood_subsampling_factor  = 0.1
    learning_rate           = 1e-5
    crop_size               = (720, 720)
    val_epoch               = num_training_epochs
    batch_size              = 8
    entropy_threshold       = 0.7


#########################################################################

class config_training_setup(object):
    """
    Setup config class for training
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args["TRAINSET"] is not None:
            self.TRAINSET = args["TRAINSET"]
        else:
            self.TRAINSET = TRAINSET
        if self.TRAINSET == "Cityscapes+COCO":
            self.roots = cs_coco_roots
            self.dataset = CityscapesCocoMix
        else:
            print("TRAINSET not correctly specified... bye...")
            exit()
        if args["MODEL"] is not None:
            tmp = getattr(self.roots, "model_name")
            roots_attr = [attr for attr in dir(self.roots) if not attr.startswith('__')]
            for attr in roots_attr:
                if tmp in getattr(self.roots, attr):
                    rep = getattr(self.roots, attr).replace(tmp, args["MODEL"])
                    setattr(self.roots, attr, rep)
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.weights_dir]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)


class config_evaluation_setup(object):
    """
    Setup config class for evaluation
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args["VALSET"] is not None:
            self.VALSET = args["VALSET"]
        else:
            self.VALSET = VALSET
        if self.VALSET == "LostAndFound":
            self.roots = laf_roots
            self.dataset = LostAndFound
        if self.VALSET == "Fishyscapes":
            self.roots = fs_roots
            self.dataset = Fishyscapes
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.io_root]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)

