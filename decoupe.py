# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:11:39 2018

@author: veylonni
"""

import os
import numpy as np
from PIL import Image

n = 2  # nombre d'imagettes sur une colonne
p = 2  # nombre d'imagettes sur une ligne

im_dir = os.path.join("images", "a_decouper")

fileNames = [f for f in os.listdir(os.path.join(im_dir)) if ".png" in f]

for f in fileNames:
    with Image.open(os.path.join(im_dir, f)) as im:
        im = np.array(im)
        if (np.array(im.shape[:2]) % [n,p]).any():
            raise ValueError("La dimension des images doit être multiple du nombre d'imagettes selon chaque axe")
        
        shape = np.array(im.shape[:2]) // [n,p]
        imagettes_dir = os.path.join(im_dir, "imagettes_{1}x{0}".format(*shape))
                
        if os.path.exists(imagettes_dir):
            raise OSError("dossier {0} existe déjà".format(imagettes_dir))
        os.makedirs(imagettes_dir)
        
        # Création des imagettes
        for i in range(n):
            for j in range(p):
                imagette = im[i*shape[0] : (i+1)*shape[0], j*shape[1] : (j+1)*shape[1]]
                imagette = Image.fromarray(imagette)
                imagette.save(os.path.join(imagettes_dir, f.split('.')[0] + "_{0}{1}.png".format(i,j)))
        