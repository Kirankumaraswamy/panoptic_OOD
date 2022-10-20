
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "./models/image-segmentation/panoptic_deeplab_model_final_23d03a.pkl"
#init_ckpt = "./weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "./config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
init_ckpt = "./models/image-segmentation/Detectron_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "./config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"'''


ood_dataset_path = "/home/mohan/kiran/cityscapes_ood"
ood_split = "val"

# whether to save the evaluation result images
save_results = False

# greater than this is OOD
#threshold = 0.1
threshold = 1
