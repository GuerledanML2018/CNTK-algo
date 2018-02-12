import numpy as np
import os
import sys
import xml.etree.cElementTree as et
import xml.dom.minidom
import cv2

def findMetaData(foldername):
    with open(foldername + "/train.txt", 'r') as mapFile:
        dataLine = mapFile.readline().split("\t")
        image = cv2.imread(dataLine[0], cv2.IMREAD_COLOR)
        imgShape = image.shape[:2]
        nbImages = len(mapFile.readlines())
        return imgShape, nbImages

def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgShape[1])
    et.SubElement(root, 'Col').text = str(imgShape[0])
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgShape[0] * imgShape[1] * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgShape[0] * imgShape[1] * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))


def computeMean(foldername):
    dataMean = np.zeros((*imgShape, 3))
    nbImages = 0
    with open(foldername + "/train.txt", 'r') as mapFile:
        for line in mapFile:
            dataLine = line.split("\t")
            image = cv2.imread(dataLine[0], cv2.IMREAD_COLOR)
            dataMean += image
            nbImages += 1
    dataMean = dataMean / nbImages
    saveMean(foldername + '/mean.xml', dataMean)



if __name__ == '__main__':
    foldername = ''.join(sys.argv[1:])
    imgShape, nbImages = findMetaData(foldername)
    computeMean(foldername)
	
