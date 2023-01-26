
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "/home/kumarasw/Thesis/models/Detectron_Panoptic_DeepLab_model_hpm_bce_loss_with_void_300.pth"
#init_ckpt = "/home/kumarasw/Thesis/Detectron_Panoptic_DeepLab_model_hpm_bce_lossi_no_void_no_ood_300.pth"
config_file = "./config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

#threshold value to use as cutoff value for OOD and In distribution
ood_threshold = 0.06

evaluate_ood=True
semantic_only = False
evaluate_anomoly = True
batch_size = 1

ood_dataset_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"
#ood_dataset_path = "/export/kiran/cityscapes"
#ood_dataset_path = "/home/kumarasw/OOD_dataset/bdd/bdd_filtered/filtered_bdd_filtered/bdd_ood"
ood_split = "val"

# whether to save the evaluation result images  partial_ood_training_0.8_0.2_324
save_results = False

result_path = "/home/kumarasw/Thesis/ksithij_results"