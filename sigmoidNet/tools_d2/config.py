import os

dataset="cityscapes"
#dataset_path = "/home/mohan/kiran/cityscapes_ood"
#dataset_path = "/home/mohan/kiran/cityscapes_ood"


split = "train"
detectron_config_file_path = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
model_name="Detectron_Panoptic_DeepLab"

suffix="sigmoid_xception_1024_2048_with_void"
start_epoch = 0
training_epoch =500
weights_dir = "./weights/"+suffix
learning_rate = 1e-5
batch_size = 1
#ckpt_path = "/home/kumarasw/Thesis/panoptic_OOD/meta-ood/weights/panoptic_deeplab_model_final_23d03a.pkl"
ckpt_path = None
#ckpt_path = "./weights/hpm_bce_loss_with_void/Detectron_Panoptic_DeepLab_model_hpm_bce_loss_with_void_300.pth"
#ckpt_path = "./weights/hpm_bce_lossi_no_void_no_ood/Detectron_Panoptic_DeepLab_model_hpm_bce_lossi_no_void_no_ood_300.pth"
#ckpt_path = "./weights/detectron_bdd/Detectron_Panoptic_DeepLab_model_detectron_bdd_200.pth"
#ckpt_path = "./weights/bdd_detectron_xception_300/Detectron_Panoptic_DeepLab_model_bdd_detectron_xception_300_60.pth"
no_gpus = 1

#os.environ["CUDA_VISIBLE_DEVICES"] ="2"

#os.environ["CITYSCAPES_DATASET"] ="/home/mohan/kiran/cityscapes_ood"

if dataset=="bdd":
    dataset_path = "/home/mohan/kiran/dataset/bdd/bdd"

    os.environ["DETECTRON2_DATASETS"] = "/home/mohan/kiran/dataset/bdd/bdd"
    os.environ["CITYSCAPES_DATASET"] ="/home/mohan/kiran/dataset/bdd/bdd/cityscapes"
else:
    dataset_path="/home/mohan/kiran/dataset/cityscapes"
    #dataset_path="/home/mohan/datasets/cityscapes"

    #dataset_path = "/home/mohan/kiran/cityscapes_ood"

    #os.environ["DETECTRON2_DATASETS"] = "/home/mohan/kiran/dataset"
    #os.environ["CITYSCAPES_DATASET"] ="/home/mohan/kiran/dataset/cityscapes"
    
    #os.environ["DETECTRON2_DATASETS"] = "/home/mohan/kiran/cityscapes_ood"
    #os.environ["CITYSCAPES_DATASET"] ="/home/mohan/kiran/cityscapes_ood/cityscapes"
    os.environ["DETECTRON2_DATASETS"] ="/home/mohan/kiran/bdd_ood"

statistics_file_name="partial_training_hpm_bce_without_void_0_8_0_2"
statistics_file_name="partial_training_hpm_bce_with_void_0_8_0_2_308"

eval_only = True
