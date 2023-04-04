import bdlb
import argparse
import os
from PIL import Image
from estimator import AnomalyDetector
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
if __name__ == '__main__':
    
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=tf_config)

    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    fs.download_and_prepare('LostAndFound')
    # fs.download_and_prepare('Static')
    # automatically downloads the dataset

    ds = tfds.load('fishyscapes/LostAndFound', split='validation')

    detector = AnomalyDetector(True)
    metrics = fs.evaluate(detector.detectron_estimator_worker, ds)

    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% FPR@95TPR'.format(100 * metrics['FPR@95%TPR']))
    print('My method achieved {:.2f}% auroc'.format(100 * metrics['auroc']))



