import bdlb
import argparse
import os
from PIL import Image
from estimator import AnomalyDetector
import numpy as np

if __name__ == '__main__':
    
    # define fishyscapes test parameters
    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    # automatically downloads the dataset
    data = fs.get_dataset('LostAndFound')
    detector = AnomalyDetector(True)
    metrics = fs.evaluate(detector.estimator_worker, data)

    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))




