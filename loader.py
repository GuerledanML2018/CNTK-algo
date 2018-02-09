# Import the relevant modules to be used later
from __future__ import print_function
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import sys

try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
    
    
# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            #labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            #print("labels",labels)
            for row in ndarray:
                row_str = row.astype(str)
                #label_str = labels[row[-1]]
                label_str = '1 0 0 0 0 0 0 0 0 0'
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)
        
        
        
def loadData(gzfname, cimg):
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 190 or ccol != 280:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        pass
#        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))


def loadLabels(gzfname, cimg):
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        pass
#        os.remove(gzfname)
    return res.reshape((cimg, 1))



        
        
if __name__ == '__main__':
    
    data_dir = os.path.join("DataSets", "test")
    gzfname = "train.gz"
    num_samples = 7
    
    data = loadData(os.path.join(data_dir, gzfname), num_samples)
    
    
    
    
#    labels = loadLabels(labelsSrc, num_samples)
#    train = np.hstack((data, labels))
    train = data
    # Plot an image
    sample_number = 0
    plt.imshow(train[sample_number,:].reshape(190,280), cmap="gray_r")
    plt.axis('off')
    print("Image Label: ", train[sample_number,-1])
        
    
    print ('Writing text file...')
    savetxt(os.path.join(data_dir, "train"), train)
    
    print('Done')