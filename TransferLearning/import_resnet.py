from __future__ import print_function
import glob
from PIL import Image
# Some of the flowers data is stored as .mat files
from shutil import copyfile
import sys
import tarfile
import os, time
import numpy as np

# Loat the right urlretrieve based on python version
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import zipfile


# By default, we store data in the Examples/Image directory under CNTK
# If you're running this _outside_ of CNTK, consider changing this
data_root = os.path.join('.')

datasets_path = os.path.join(data_root, 'DataSets')
output_path = os.path.join('.', 'Output')

def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(file_path, img_paths, img_labels):
    with open(file_path, 'w+') as f:
        for i in range(0, len(img_paths)):
            f.write('%s\t%s\n' % (os.path.abspath(img_paths[i]), img_labels[i]))

def download_unless_exists(url, filename, max_retries=3):
    '''Download the file unless it already exists, with retry. Throws if all retries fail.'''
    if os.path.exists(filename):
        print('Reusing locally cached: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        retry_cnt = 0
        while True:
            try:
                urlretrieve(url, filename)
                print('Download completed.')
                return
            except:
                retry_cnt += 1
                if retry_cnt == max_retries:
                    print('Exceeded maximum retry count, aborting.')
                    raise
                print('Failed to download, retrying.')
                time.sleep(np.random.randint(1,10))

def download_model(model_root = os.path.join(data_root, 'PretrainedModels')):
    ensure_exists(model_root)
    resnet18_model_uri = 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model'
    resnet18_model_local = os.path.join(model_root, 'ResNet_18.model')
    download_unless_exists(resnet18_model_uri, resnet18_model_local)
    return resnet18_model_local

download_model()
