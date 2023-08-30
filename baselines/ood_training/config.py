cityscapes_ood_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes_ood"
cityscapes_eval_path = "/home/kumarasw/OOD_dataset/filtered_OOD_dataset/cityscapes_ood"
split = "train"
detectron_config_file_path = "/home/kumarasw/Thesis/panoptic_OOD/ood_training/config/panopticDeeplab/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
model_name="Detectron_Panoptic_DeepLab"

start_epoch = 0
training_epoch =4
weights_dir = "/home/kumarasw/Thesis/panoptic_OOD/ood_training/weights"
learning_rate = 1e-5
batch_size = 1
ckpt_path = "/home/kumarasw/Thesis/panoptic_OOD/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
no_gpus = 2
