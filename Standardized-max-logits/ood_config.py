
model_name = "Detectron_Panoptic_DeepLab"
init_ckpt = "./pretrained/panoptic_deeplab_model_final_23d03a.pkl"
config_file = "./config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"

'''model_name = "Detectron_DeepLab"
init_ckpt = "./pretrained/deeplab_model_final_a8a355.pkl"
config_file = "./config/deeplab/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
'''
'''model_name = "DeepLabV3+_WideResNet38"
init_ckpt = "./weights/DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"'''

# Whether to use soft_max output for thresholding
max_softmax = False

#threshold value to use as cutoff value for OOD and In distribution
threshold = 50

ood_dataset_path = "/home/mohan/kiran/cityscapes_ood"
ood_split = "val"

# whether to save the evaluation result images
save_results = False
