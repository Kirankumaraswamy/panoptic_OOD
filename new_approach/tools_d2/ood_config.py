import os
model_name = "Detectron_Panoptic_DeepLab"

dataset = "cityscapes"
#dataset = "bdd"
if dataset == "cityscapes":

    init_ckpt = "./experiment_results/xception_sigmoid_bce_512_1024_bs16/model_0069999.pth"
    #init_ckpt = "./experiment_results/final_experiments/cityscapes_sigmoid_69999_with_out_ood_head/model_0014879.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_no_ood_head_sem_0_1_0_9/model_0018599.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_no_ood_head_sem_0_1_0_9/model_0007439.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_3_sem_0_1_0_9/model_0018599.pth" 
    #init_ckpt = "./experiment_results/final_experiments/softmax_ood_retraining/model_0003719.pth"
    #init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/panoptic-deeplab/tools_d2/weights/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
    #init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/results_meta_ood/results_xception_cityscapes_ood/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
    #init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/resultsfinal_meta_ood_Detectron_Panoptic_DeepLab/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
    #init_ckpt = "./experiment_results/bdd_xception_sigmoid_bce_512_1024_bs16/model_0124999.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_3/model_0001859.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_2/model_0013019.pth"

    #init_ckpt = "./experiment_results/final_experiments/softmax_ood_retraining/model_0003719.pth"
    init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/panoptic-deeplab/tools_d2/weights/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.pth"
    
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_head_sem_0_1_0_9/model_0001859.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_3/model_0001859.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_2/model_0013019.pth"
    
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_3_sem_0_1_0_9/model_0007439.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/city_new/cityscapes_sigmoid_69999_with_head_05_sem_0_1_0_9/model_0001859.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/city_new/cityscapes_sigmoid_69999_with_head_scale_weights_copy_sem_0_1_0_9/model_0001859.pth"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_ood_head_0_1_0_3_sem_0_1_0_9/model_0016739.pth"
    #config_file = "./configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
    #init_ckpt = "./experiment_results/cityscapes_final_experiments/cityscapes_sigmoid_69999_with_head_sem_0_1_0_9/model_0001859.pth"
    
    config_file = "./configs/Cityscapes-PanopticSegmentation/ood_training_with_head_panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
    #threshold value to use as cutoff value for OOD and In distribution
    ood_threshold = 1.1
    
    #init_ckpt="experiment_results/void_class_training_xception_bs16_512_1024/model_0089999.pth"
    #config_file="./configs/Cityscapes-PanopticSegmentation/void_class_training_panoptic_deeplab_X_65_os16_mg124_poly_90k_bs16_crop_512_1024_dsconv.yaml"

    #ood_dataset_path = "/home/mohan/kiran/cityscapes_ood/cityscapes"
    #ood_dataset_path = "/home/mohan/kiran/dataset/cityscapes"
    ood_dataset_path = "/home/mohan/kiran/cityscapes_ood_unseen/cityscapes"
    ood_split = "val"

else:

    init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_bdd_ood_no_head_lr_00001_sem_0_1_0_9/model_0008749.pth"
    init_ckpt = "./experiment_results/bdd_final_experiments/bdd_sigmoid_69999_with_ood_head_0_1_0_3_sem_0_1_0_9_lr_0001/model_0008749.pth"
    init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_bdd_ood_head_0_1_0_3_lr_0005/model_0034999.pth"
    #model_0008749.pth,  model_0017499.pth, model_0026249.pth, model_0034999.pth
    init_ckpt = "./experiment_results/bdd_xception_sigmoid_bce_512_1024_bs16/model_0124999.pth"
    #init_ckpt = "/home/mohan/kiran/ponoptic_OOD/meta-ood/results_meta_ood/results_xception_bdd_ood/weights/Detectron_Panoptic_DeepLab_epoch_4_alpha_0.9.pth"
    #init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_ood_head_0_1_0_3_lr_00001/model_0017499.pth"
    #init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_ood_head_0_1_0_3_lr_00001_0_1_0_9_sem/model_0008749.pth"
    #init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_bdd_ood_no_head_lr_00001/model_0008749.pth"
    #init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_ood_head_new_weights_lr_00001_0_1_0_9_sem/model_0008749.pth"
    #config_file="./configs/Cityscapes-PanopticSegmentation/void_class_training_panoptic_deeplab_X_65_os16_mg124_poly_90k_bs16_crop_512_1024_dsconv.yaml" 
    config_file = "./configs/Cityscapes-PanopticSegmentation/bdd_ood_training_panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv_test.py"

    #threshold value to use as cutoff value for OOD and In distribution
    ood_threshold = 0.8
    #config_file = "configs/Cityscapes-PanopticSegmentation/bdd_panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
    #init_ckpt = "experiment_results/bdd_void_class_training_xception_bs16_512_1024/model_0109999.pth"
    #init_ckpt = "./experiment_results/bdd_final_experiments/bdd_xception_sigmoid_with_ood_head_01_03_lr_00025_sem_0_1_0_9/model_0017499.pth"
    #ood_dataset_path = "/home/mohan/kiran/dataset/bdd_new/cityscapes/"
    #ood_dataset_path="/home/mohan/kiran/bdd_ood_train/bdd_ood/cityscapes"
    ood_dataset_path="/home/mohan/kiran/bdd_ood_val/filtered_bdd_filtered/cityscapes"
    ood_split = "val"

evaluate_ood=True
semantic_only = False
evaluate_anomoly = False
performance_with_ood = False

batch_size =1 

# whether to save the evaluation result images
save_results = True

os.environ["CUDA_VISIBLE_DEVICES"] ="4"

#save_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
save_instance_path = None
read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_bdd"
#read_instance_path = "/home/mohan/kiran/center_offsets/meta_ood_cityscapes"
#read_instance_path = None
#read_semantic_path = "/home/mohan/kiran/center_offsets/ksithij_semantic"
read_semantic_path = None
