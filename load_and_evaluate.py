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

# model dimensions
image_height = 128
image_width  = 72
num_channels = 3
num_classes  = 3

def compute_confusion_matrix(pred):
    # Load the true labels
    true_labels = []
    images = []
    for l in open("Guerledan/test.txt", "r") :
        images.append(l.split('\t')[0])
        true_labels.append(int(l.split('\t')[1][:-1]))

    # evaluate all the images
    ev_labels = []
    for im_name in images:
        myimg = np.array(PIL.Image.open(im_name), dtype=np.float32).reshape((
                         image_height, image_width, num_channels))
        result = eval(pred, myimg)
        ev_labels.append(result[0])

    # compute confusion matrix
    conf_mat = metrics.confusion_matrix(true_labels, ev_labels)
    return conf_mat

def eval(pred_op, image_data):
    label_lookup = ["Rochers", "Terrains vagues", "Forêts"]
    image_mean = 133.0
    image_data -= image_mean
    image_data = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))

    result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))

    # Return top 3 results:
    top_count = 3
    result_indices = (-np.array(result)).argsort()[:top_count]

    # print("Top 3 predictions:")
    # for i in range(top_count):
    #     print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100))
    return result_indices


# image_name = sys.argv[1]
# myimg = np.array(PIL.Image.open(image_name), dtype=np.float32).reshape((
#     image_height, image_width, num_channels))

mod = "save/trainv3_1.save"
z = C.load_model(mod)
print("Modèle chargé :", mod)
pred = C.softmax(z)
# eval(pred, myimg)
conf_mat = compute_confusion_matrix(pred)
print(conf_mat)
print("Pourcentage de réussite :", 100*sum(np.diag(conf_mat))/np.sum(
    conf_mat), "%")
