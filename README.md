# CNTK-algo
Description

* `rename_images.py` et `resize.sh` sont des scripts pour faire des transformations simples sur un grand nombre d'images à la fois.
* `video_acq.py` permet d'enregistrer des images depuis une caméra. Pour géolocaliser ces images, voir le dossier `acquisition_im_pos`.
* `store_data.py` est le script d'étiquettage des données.

## acquisition_im_pos
C'est un paquet ROS qui permet de lancer l'acquisition d'images depuis une caméra et de géolocaliser ces images avec un GPS.
le fichier `acquisition.launch` lance les 2 noeuds correspondant.

## CNN
C'est le dossier comprenant les scripts permettant de créer des CNN simples, souvent de moins de 10 couches.

## TransferLearning
Ici, il y a les scripts qui permettent d'importer un modèle déjà entraîner et de réentrainer sa dernière couche pour l'adapter au problème donné. Les modèles pré entrainés ne sont pas stockés sur GitHub puisqu'ils sont trop volumineux.

## Unsupervised_learning
Ce sont nos tests pour l'apprentissage non supervisés, que nous n'avons pas terminés.
