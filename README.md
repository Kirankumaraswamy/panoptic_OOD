# Out-of-distribution detection for Panoptic segmentation

## Introduction
This is a repository for my Master's Thesis on Out-of-distribution detection for Panoptic segmentation. My Thesis document can be accessed at https://drive.google.com/file/d/1NfLQlkZF5kzgj_UifYoviU6IphbP1TT5/view?usp=drive_link

<br>
**Abstract:** Deep neural networks trained for perception tasks are often trained to identify object instances belonging to a predeőned set of classes (in-distribution). However, in the real world when these networks encounter objects that they are not trained for (out-of-distribution), it poses a safety risk in applications such as autonomous driving as these models often misclassify them as one of the in-distribution class or fail to detect them entirely. For holistic scene understanding using panoptic segmentation, identifying each individual out-of-distribution object in the scene is vital compared to identifying only their semantic class. To address this limitation, we propose a novel perception task, Panoptic Out-of-Distribution Segmentation, which involves joint pixel-level in-distribution and out-of-distribution semantic segmentation, as well as individual instance prediction. We also identify the challenges associated with this task and propose a new metric for evaluating the joint task of out-of-distribution panoptic segmentation. Additionally, we present two new datasets by extending the two existing datasets for panoptic segmentation - Cityscapes and BDD100K, by introducing numerous out-of-distribution (OOD) instances in the original images and their annotations. Our datasets incorporate dynamic scaling of out-of-distribution instances and several image post-processing methods. More importantly, we propose a modiőed training pipeline for a bottom-up panoptic segmentation network by utilizing sigmoid activation as a őnal output layer for semantic segmentation head trained with binary cross-entropy loss and two additional OOD heads for instance
segmentation. Our modiőed network further with retraining on our new dataset establishes a generalized out-of-distribution detection capability while maintaining the in-distribution panoptic segmentation performance. Furthermore, we present multiple baselines and perform extensive evaluations to demonstrate that our proposed approach addresses all the challenges and outperforms the evaluated baselines on our new task.


## Folder structure
<b>sigmoidNet</b>: Consists code to reproduce our proposed architecture utilizing panoptic deeplab to detect anomolous objects in the autonomous driving scene.<br>
<b>metric</b>: Consists benchmark scripts to evaluate our new proposed task "Panoptic out-of-distribution segmentation". The scripts are established utilizing existing standard evaluation scripts for panoptic segmentation Cityscapesscripts and Panopticapi.<br>
<b>baselines</b>: Consists of several baselines to detect anomolous objects in panoptic segmentation. These baselines were reproduced by extending existing approches for detecting anomolous objects in a semantic segmentation scenario.<br>
<b>panoptic_ood_dataset</b>: Consists data creation pipeline for two new datasets BDD100k-OOD and Cityscapes-OOD which were created to evaluate our new task.

## Installation
Since the baselines and sigmoidNet were reproduced with detectron2 version of panoptic-deeplab, follow detectron2 installation [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).  
