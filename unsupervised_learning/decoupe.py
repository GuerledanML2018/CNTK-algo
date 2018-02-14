# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:11:39 2018

@author: veylonni
"""

import os
import numpy as np
from PIL import Image

nb_ligne = 2
nb_colonnes = 4
overlapping = 0.0 # en %

im_dir = os.path.join("..", "..",
                      "data_guerledan_metz_dangers_sans_ambiguite", "train",
                      "test_decoupe")

fileNames = [f for f in os.listdir(os.path.join(im_dir)) if ".jpg" in f or
             ".jpeg" in f]
# np.random.shuffle(fileNames)  # TODO :
# fileNames = fileNames[:50]  #TODO :

nb_images = len(fileNames)
im_traitees = 0

for i, f in enumerate(fileNames):
    with Image.open(os.path.join(im_dir, f)) as im:
        im = np.array(im)
        if (np.array(im.shape[:2]) % [nb_ligne, nb_colonnes]).any():
            raise ValueError("La dimension des images doit être multiple du nombre d'imagettes selon chaque axe")

        # shape_imagette = np.array(im.shape[:2]) // [nb_ligne, nb_colonnes]
        shape_imagette = np.array(im.shape[:2]) // [(nb_ligne - (nb_ligne-1) * overlapping),
                                                    (nb_colonnes - (nb_colonnes-1) * overlapping)]
        shape_imagette = np.array(shape_imagette, dtype=np.int)
        imagettes_dir = os.path.join(im_dir, "imagettes_{1}x{0}".format(*shape_imagette))

        pix_overlapping = np.array([(nb_ligne*shape_imagette[0] -
                                         im.shape[0]) / (nb_ligne - 1),
                                    (nb_colonnes*shape_imagette[1] -
                                         im.shape[1]) / (nb_colonnes - 1)])

        if not os.path.exists(imagettes_dir):
            os.makedirs(imagettes_dir)

        # Création des imagettes

        # imagette 0,0
        imagette = im[0 : shape_imagette[0],
                   0 : shape_imagette[1]]
        imagette = Image.fromarray(imagette)
        imagette.save(os.path.join(imagettes_dir, f.split('.')[0] + "_{0}{1}.png".format(0,0)))

        # imagettes 0, j
        for j in range(1, nb_colonnes):
            imagette = im[0 : shape_imagette[0],
                       j * shape_imagette[1] - int(j * pix_overlapping[1]):
                       (j + 1) * shape_imagette[1] - int(j * pix_overlapping[1])]
            imagette = Image.fromarray(imagette)
            imagette.save(os.path.join(imagettes_dir,
                                       f.split('.')[0] + "_{0}{1}.png".format(0, j)))

        # imagettes centrales
        for i in range(1, nb_ligne):
            imagette = im[i * shape_imagette[0] - int(i * pix_overlapping[0]) :
                              (i+1)*shape_imagette[0] - int(i *pix_overlapping[0]),
                               0 : shape_imagette[1]]
            imagette = Image.fromarray(imagette)
            imagette.save(os.path.join(imagettes_dir,
                                       f.split('.')[0] + "_{0}{1}.png".format(i, 0)))
            for j in range(1, nb_colonnes):
                # imagette = im[i*shape_imagette[0] : (i+1)*shape_imagette[0], j*shape_imagette[1] : (j+1)*shape_imagette[1]]
                imagette = im[i * shape_imagette[0] - int(i * pix_overlapping[0]) :
                              (i+1)*shape_imagette[0] - int(i *pix_overlapping[0]),
                           j * shape_imagette[1] - int(j * pix_overlapping[1]):
                           (j + 1) * shape_imagette[1] - int(j * pix_overlapping[1])]
                imagette = Image.fromarray(imagette)
                imagette.save(os.path.join(imagettes_dir, f.split('.')[0] + "_{0}{1}.png".format(i,j)))


    im_traitees += 1
    print(im_traitees, "images traitées sur", nb_images, "(",
          (im_traitees)/nb_images*100, "%)")

print("Terminé")
