
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "/home/kumarasw/Thesis/Detectron_Panoptic_DeepLab_best_model_hard_pixel_mining_bce_loss.pth"
config_file = "./config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

#threshold value to use as cutoff value for OOD and In distribution
threshold = 0.1

#ood_dataset_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"
ood_dataset_path = "/export/kiran/cityscapes"
ood_split = "val"

# whether to save the evaluation result images
save_results = True

class_mean_path = './stats/cityscapes_panoptic_deeplab_hard_pixel_mining_mean.npy'
class_var_path = './stats/cityscapes_panoptic_deeplab_hard_pixel_mining_var.npy'
