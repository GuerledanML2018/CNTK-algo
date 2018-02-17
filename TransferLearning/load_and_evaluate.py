from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import matplotlib.pyplot as plt
import numpy as np
import PIL
import sys
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

import cntk as C
import sklearn.metrics as metrics

from PIL import Image

DATASET_NAME = 'data_guerledan_metz_dangers'
SAVEFILE_NAME = 'guerledan_save_5.model'

# model dimensions
image_height = 128
image_width  = 72
num_channels = 3
num_classes  = 2

def compute_confusion_matrix(pred, tolerance = 0.9):
    # Load the true labels
    true_labels = []
    images = []
    for l in open("../../" + DATASET_NAME + "/test.txt", "r") :
        images.append(l.split('\t')[0])
        true_labels.append(int(l.split('\t')[1][:-1]))

    # evaluate all the images
    # nb_err = 0
    # vp = []
    ev_labels = []
    for indice_im, im_name in enumerate(images):
        result = eval_single_image(pred, im_name, (3, 224, 224))
        label = 0 if result[0] >= tolerance else 1
        ev_labels.append(label)
        print("Image", indice_im, "sur", len(images), "-->", round(indice_im/len(images)*100, 2), "%")
        # if label != true_labels[indice_im]:
            # if label == 0:
            #     vp.append(im_name)
            # nb_err += 1
            # print(im_name, label, result, true_labels[indice_im])
    # print("Nombre d'erreurs : ", nb_err)
    # with open("vp.txt", 'w') as f:
    #     for v in vp:
    #         f.write(v)
    print("Terminé\n")

    # compute confusion matrix
    conf_mat = metrics.confusion_matrix(true_labels, ev_labels)
    return conf_mat


def eval_single_image(loaded_model, image_path, image_dims):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    try:
        img = Image.open(image_path)

        if image_path.endswith("png"):
            temp = Image.new("RGB", img.size, (255, 255, 255))
            temp.paste(img, img)
            img = temp
        resized = img.resize((image_dims[2], image_dims[1]), Image.ANTIALIAS)
        bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {loaded_model.arguments[0]: [hwc_format]}
        output = loaded_model.eval(arguments)

        # return softmax probabilities
        sm = C.softmax(output[0])
        return sm.eval()
    except FileNotFoundError:
        print("Could not open (skipping file): ", image_path)
        return ['None']


if __name__ == "__main__":

    mod = "temp/Output/" + SAVEFILE_NAME
    z = C.load_model(mod)
    print("Modèle chargé :", mod)
    # pred = C.softmax(z)


    # eval(pred, myimg)
    conf_mat = compute_confusion_matrix(z)
    print(conf_mat)
    print("Pourcentage de réussite :", 100*sum(np.diag(conf_mat))/np.sum(
        conf_mat), "%")