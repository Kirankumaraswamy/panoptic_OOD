
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "/home/kumarasw/Thesis/panoptic_OOD/meta-ood/panoptic-deeplab/tools_d2/weights/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
config_file = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
init_ckpt = "./pretrained/deeplab_model_final_a8a355.pkl"
config_file = "./config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
'''
'''model_name = "DeepLabV3+_WideResNet38"
init_ckpt = "./weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''


#threshold value to use as cutoff value for OOD and In distribution
ood_threshold = 0.7
evaluate_ood=True
semantic_only = False
evaluate_anomoly = True
batch_size = 1


ood_dataset_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"
#ood_dataset_path = "/export/kiran/cityscapes"
ood_split = "val"

# whether to save the evaluation result images
save_results = True
