import os, sys

sys.path.append(os.path.join('..', 'unsupervised_learning'))
from unsupervised_learning.decoupe import decoupe

nb_lignes = 1
nb_colonnes = 4
overlapping = 0.3456

root_im_dir = os.path.join('..', '..', 'data_guerledan_metz_dangers_sans_ambiguite_decoupees')

with open(os.path.join(root_im_dir, "data.txt"), 'w') as fw:
    with open(os.path.join(root_im_dir, "data_old.txt"), 'r') as fr:
        for l in fr:
            im_path = l.split()[0]
            label = l.split()[1]

            imagettes_names = decoupe(nb_lignes, nb_colonnes, overlapping, im_path, os.path.join(root_im_dir, 'imagettes'))

            for names in imagettes_names:
                fw.write((names + ' ' + label + '\n') if label == '1' else '')


print("Terminé")