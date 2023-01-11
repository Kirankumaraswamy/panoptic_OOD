import os

dataset_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"

dataset="cityscapes"

split = "train"
detectron_config_file_path = "/home/kumarasw/Thesis/panoptic_OOD/ood_training/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
model_name="Detectron_Panoptic_DeepLab"

start_epoch = 0
training_epoch =300

suffix = "default_crop"

weights_dir = "/home/kumarasw/Thesis/panoptic_OOD/ood_training/weights"
learning_rate = 1e-5
batch_size = 1
ckpt_path = "/home/kumarasw/Thesis/panoptic_OOD/meta-ood/weights/Detectron_Panoptic_DeepLab_best_model_uncertainity_head.pth"
ckpt_path = None
ckpt_path = "/home/kumarasw/Thesis/models/Detectron_Panoptic_DeepLab_model_hpm_bce_loss_with_void_300.pth"
no_gpus = 2

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
#os.environ["DETECTRON2_DATASETS"] = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset"
#os.environ["CITYSCAPES_DATASET"] ="/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"


if dataset=="bdd":
    dataset_path = "/home/kumarasw/OOD_dataset/bdd/bdd"

    os.environ["DETECTRON2_DATASETS"] = "/home/mohan/kiran/dataset"
    os.environ["CITYSCAPES_DATASET"] ="/home/mohan/kiran/dataset/bdd"
else:
    dataset_path="/export/kiran/cityscapes"
    dataset_path="/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"

    os.environ["DETECTRON2_DATASETS"] = "/export/kiran"
    os.environ["CITYSCAPES_DATASET"] ="/export/kiran/cityscapes"
    os.environ["CITYSCAPES_DATASET"] = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"


statistics_file_name = "pn_training_no_hpm_0_25_120"

sum_all_class_mean_path = f'./stats/sum_all_{statistics_file_name}_mean.npy'
sum_all_class_var_path = f'./stats/sum_all_{statistics_file_name}_var.npy'

#sum_all_class_mean_path = f'./stats/cityscapes_panoptic_deeplab_hard_pixel_mining_mean.npy'
#sum_all_class_var_path = f'./stats/cityscapes_panoptic_deeplab_hard_pixel_mining_var.npy'

correct_class_mean_path = f'./stats/correct_class_{statistics_file_name}_mean.npy'
correct_class_var_path = f'./stats/correct_class_{statistics_file_name}_var.npy'
sum_non_class_mean_path = f'./stats/sum_non_class_{statistics_file_name}_mean.npy'
sum_non_class_var_path = f'./stats/sum_non_class_{statistics_file_name}_var.npy'

class_mean = f'./stats/in_dist_neg_{statistics_file_name}_mean.npy'
class_var = f'./stats/in_dist_neg_{statistics_file_name}_var.npy'
