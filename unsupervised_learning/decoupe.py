# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:11:39 2018

@author: veylonni
"""

import os
import numpy as np
from PIL import Image


def decoupe(nb_lignes, nb_colonnes, overlapping, im_path, output_dir):
    im_names = []
    with Image.open(os.path.join(im_path)) as im:
        im = np.array(im)
        if (np.array(im.shape[:2]) % [nb_lignes, nb_colonnes]).any():
            raise ValueError("La dimension des images doit être multiple du nombre d'imagettes selon chaque axe")

        # shape_imagette = np.array(im.shape[:2]) // [nb_ligne, nb_colonnes]
        shape_imagette = np.array(im.shape[:2]) // [(nb_lignes - (nb_lignes - 1) * overlapping),
                                                    (nb_colonnes - (nb_colonnes-1) * overlapping)]
        shape_imagette = np.array(shape_imagette, dtype=np.int)
        imagettes_dir = os.path.join(output_dir)

        pix_overlapping = np.array([(nb_lignes * shape_imagette[0] -
                                     im.shape[0]) / (nb_lignes - 1),
                                    (nb_colonnes*shape_imagette[1] -
                                         im.shape[1]) / (nb_colonnes - 1)])

        if not os.path.exists(imagettes_dir):
            os.makedirs(imagettes_dir)

        # Création des imagettes

        # imagette 0,0
        imagette = im[0 : shape_imagette[0],
                   0 : shape_imagette[1]]
        imagette = Image.fromarray(imagette)
        im_names.append(os.path.join(output_dir, os.path.split(im_path)[1].split('.')[0] + "_{0}{1}.jpg".format(0,0)))
        imagette.save(im_names[-1])

        # imagettes 0, j
        for j in range(1, nb_colonnes):
            imagette = im[0 : shape_imagette[0],
                       j * shape_imagette[1] - int(j * pix_overlapping[1]):
                       (j + 1) * shape_imagette[1] - int(j * pix_overlapping[1])]
            imagette = Image.fromarray(imagette)
            im_names.append(os.path.join(output_dir, os.path.split(im_path)[1].split('.')[0] + "_{0}{1}.jpg".format(0, j)))
            imagette.save(im_names[-1])

        # imagettes centrales
        for i in range(1, nb_lignes):
            imagette = im[i * shape_imagette[0] - int(i * pix_overlapping[0]) :
                              (i+1)*shape_imagette[0] - int(i *pix_overlapping[0]),
                               0 : shape_imagette[1]]
            imagette = Image.fromarray(imagette)
            im_names.append(os.path.join(output_dir, os.path.split(im_path)[1].split('.')[0] + "_{0}{1}.jpg".format(i, 0)))
            imagette.save(im_names[-1])
            for j in range(1, nb_colonnes):
                # imagette = im[i*shape_imagette[0] : (i+1)*shape_imagette[0], j*shape_imagette[1] : (j+1)*shape_imagette[1]]
                imagette = im[i * shape_imagette[0] - int(i * pix_overlapping[0]) :
                              (i+1)*shape_imagette[0] - int(i *pix_overlapping[0]),
                           j * shape_imagette[1] - int(j * pix_overlapping[1]):
                           (j + 1) * shape_imagette[1] - int(j * pix_overlapping[1])]
                imagette = Image.fromarray(imagette)
                im_names.append(os.path.join(output_dir, os.path.split(im_path)[1].split('.')[0] + "_{0}{1}.jpg".format(i, j)))
                imagette.save(im_names[-1])


    # im_traitees += 1
    # print(im_traitees, "images traitées sur", nb_images, "(",
    #       (im_traitees)/nb_images*100, "%)")

    # print("Terminé")

    return im_names

if __name__ == "__main__":
    nb_lignes = 1
    nb_colonnes = 8
    overlapping = 0.75  # en % / 100

    im_dir = os.path.join("..", "..", "images_test", "0011.jpg")
    output_dir = os.path.join("..", "..", "images_test")

    decoupe(nb_lignes, nb_colonnes, overlapping, im_dir, output_dir)
