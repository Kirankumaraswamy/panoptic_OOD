import os

model_name = "Detectron_Panoptic_DeepLab"
#init_ckpt = "./weights/panoptic_deeplab_model_final_23d03a.pkl"
#init_ckpt = "./weights/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
#init_ckpt = "../../results_meta_ood/results_xception_cityscapes_ood/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
#init_ckpt = "../../results_meta_ood/results_xception_bdd_ood/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
init_ckpt = "./weights/bdd_xception_bs16_512_1024_0124999.pth"

#init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/resultsfinal_meta_ood_Detectron_Panoptic_DeepLab/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
cfg.merge_from_file("./src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
init_ckpt = "./weights/Detectron_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "../src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"'''

'''model_name = "DeepLabV3+_WideResNet38"
init_ckpt = "./weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''

# Whether to use soft_max output for thresholding
max_softmax = False

#threshold value to use as cutoff value for OOD and In distribution
# for entropy the value above this is classified as OOD
# for softmax the value below this is used as OOD
# If the value is 1 for entropy, it means the model doesn't identifies OOD. It is a regular Detectron model
ood_threshold = 0.35

evaluate_ood = True
semantic_only = False
evaluate_anomoly = False
performance_with_ood = False
batch_size = 1

#os.environ["CUDA_VISIBLE_DEVICES"] ="6"

#ood_dataset_path = "/home/mohan/kiran/cityscapes_ood/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/dataset/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/cityscapes_ood_unseen/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_original/bdd_new/cityscapes/"
ood_dataset_path = "/home/mohan/kiran/bdd_ood_val/filtered_bdd_filtered/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_ood_train/bdd_ood/cityscapes/"
ood_split = "val"

# whether to save the evaluation result images
save_results = False

#save_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
save_instance_path = None
read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
#read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_cityscapes"
#read_instance_path = None
#read_semantic_path = "/home/mohan/kiran/center_offsets/ksithij_semantic_cityscapes"
read_semantic_path = "/home/mohan/kiran/center_offsets/ksithij_semantic_bdd"
#read_semantic_path = None
