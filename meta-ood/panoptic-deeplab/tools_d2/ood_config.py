
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "/home/kumarasw/Thesis/panoptic_OOD/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
#init_ckpt = "./weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "./src/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
cfg.merge_from_file("./src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
init_ckpt = "./weights/Detectron_DeepLab_epoch_4_alpha_0.9.pth"
config_file = "../src/config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"'''

'''model_name = "DeepLabV3+_WideResNet38"
init_ckpt = "./weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''

# Whether to use soft_max output for thresholding
max_softmax = True

#threshold value to use as cutoff value for OOD and In distribution
# for entropy the value above this is classified as OOD
# for softmax the value below this is used as OOD
# If the value is 1 for entropy, it means the model doesn't identifies OOD. It is a regular Detectron model
ood_threshold = 1

evaluate_ood=True
semantic_only = False
evaluate_anomoly = False
batch_size = 1

ood_dataset_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes"
ood_split = "val"

# whether to save the evaluation result images
save_results = False
