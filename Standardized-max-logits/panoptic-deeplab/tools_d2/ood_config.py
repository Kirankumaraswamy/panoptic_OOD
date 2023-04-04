
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "../../pretrained/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
#init_ckpt = "../../pretrained/bdd_xception_bs16_512_1024_0124999.pth"
config_file = "configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
#init_ckpt="../../pretrained/panoptic_deeplab_model_final_23d03a.pkl"
#config_file = "configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
init_ckpt = "./pretrained/deeplab_model_final_a8a355.pkl"
config_file = "./config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
'''
'''model_name = "DeepLabV3+_WideResNet38"
init_ckpt = "./weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''


#threshold value to use as cutoff value for OOD and In distribution
ood_threshold = 0.2
evaluate_ood=True
semantic_only = False
evaluate_anomoly = False
performance_with_ood = False
batch_size = 1

#ood_dataset_path = "/home/mohan/kiran/cityscapes_ood/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/dataset/cityscapes"
ood_dataset_path = "/home/mohan/kiran/cityscapes_ood_unseen/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_original/bdd_new/cityscapes/"
#ood_dataset_path = "/home/mohan/kiran/bdd_ood_val/filtered_bdd_filtered/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_ood_train/bdd_ood/cityscapes/"

ood_split = "val"

# whether to save the evaluation result images
save_results = False


class_mean = "./stats/cityscapes_Detectron_Panoptic_DeepLab_mean.npy"
class_var = "./stats/cityscapes_Detectron_Panoptic_DeepLab_var.npy"

#class_mean = "./stats/bdd_Detectron_Panoptic_DeepLab_mean.npy"
#class_var = "./stats/bdd_Detectron_Panoptic_DeepLab_var.npy"

#save_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
save_instance_path = None
#read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_cityscapes"
#read_instance_path = None
#read_semantic_path = "/home/mohan/kiran/center_offsets/ksithij_semantic"
read_semantic_path = None

