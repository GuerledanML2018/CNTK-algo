from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import matplotlib.pyplot as plt
import numpy as np
import sys, os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

import cntk as C
import sklearn.metrics as metrics

from PIL import Image
from TransferLearning.load_and_evaluate import eval_single_image
sys.path.append(os.path.join('..', 'unsupervised_learning'))
from unsupervised_learning.decoupe import decoupe

DATASET_NAME = 'data_guerledan_metz_dangers_sans_ambiguite_decoupees'
SAVEFILE_NAME = 'guerledan_save_10.model'

# model dimensions
image_height = 128
image_width  = 72
num_channels = 3
num_classes  = 2

output_path = os.path.join('temp', 'imagettes')
if not os.path.exists(output_path):
    os.makedirs(output_path)

def compute_confusion_matrix(pred):
    # Load the true labels
    true_labels = []
    images = []
    for l in open("../../" + DATASET_NAME + "/validation.txt", "r") :
        images.append(l.split('\t')[0])
        true_labels.append(int(l.split('\t')[1][:-1]))

    # evaluate all the images
    ev_labels = []
    for indice_im, im_name in enumerate(images):
        result = eval_single_full_image(pred, im_name, (3, 224, 224), tolerance=0.5)
        label = result
        ev_labels.append(label)
        print("Image", indice_im, "sur", len(images), "-->", round(indice_im/len(images)*100, 2), "%")
    print("Terminé\n")

    # compute confusion matrix
    conf_mat = metrics.confusion_matrix(true_labels, ev_labels)
    return conf_mat


def eval_single_full_image(loaded_model, image_path, image_dims, tolerance):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    try:
        imagette_names = decoupe(8, 1, 0.75, image_path, output_path)
        os.remove(imagette_names[1])
        os.remove(imagette_names[3])
        os.remove(imagette_names[5])
        os.remove(imagette_names[6])
        imagette_names = [imagette_names[0], imagette_names[2], imagette_names[4], imagette_names[7]]

        probas = []
        for imagette in imagette_names:
            img = Image.open(imagette)
            resized = img.resize((image_dims[2], image_dims[1]), Image.ANTIALIAS)
            bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
            hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

            # compute model output
            arguments = {loaded_model.arguments[0]: [hwc_format]}
            output = loaded_model.eval(arguments)

            # return softmax probabilities
            probas.append(0 if C.softmax(output[0]).eval()[0] >= tolerance else 1)
            os.remove(imagette)

        return int(np.array(probas).all())
    except FileNotFoundError:
        print("Could not open (exiting): ", image_path)
        exit(1)



mod = "temp/Output/" + SAVEFILE_NAME
z = C.load_model(mod)
print("Modèle chargé :", mod)
# pred = C.softmax(z)


# eval(pred, myimg)
conf_mat = compute_confusion_matrix(z)
print(conf_mat)
print("Pourcentage de réussite :", 100*sum(np.diag(conf_mat))/np.sum(
    conf_mat), "%")