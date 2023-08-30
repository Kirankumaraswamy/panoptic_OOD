
model_name = "Detectron_Panoptic_DeepLab"
#init_ckpt = "./models/image-segmentation/panoptic_deeplab_model_final_23d03a.pkl"
#config_file = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

init_ckpt = "./models/image-segmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
config_file = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
init_ckpt = "./models/image-segmentation/Detectron_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "./config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"'''

#ood_dataset_path = "/home/mohan/kiran/cityscapes_ood/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/dataset/cityscapes"
ood_dataset_path = "/home/mohan/kiran/cityscapes_ood_unseen/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_original/bdd_new/cityscapes/"
#ood_dataset_path = "/home/mohan/kiran/bdd_ood_val/filtered_bdd_filtered/cityscapes"
#ood_dataset_path = "/home/mohan/kiran/bdd_ood_train/bdd_ood/cityscapes/"
ood_split = "val"

# whether to save the evaluation result images
save_results = False

# greater than this is OOD
#threshold = 0.1
ood_threshold = 0.7
evaluate_ood=True
semantic_only = False
evaluate_anomoly = False
performance_with_ood = True
batch_size = 1

#save_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
save_instance_path = None
read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_cityscapes"
#read_instance_path = None
#read_semantic_path = "/home/mohan/kiran/center_offsets/ksithij_semantic"
read_semantic_path = None
