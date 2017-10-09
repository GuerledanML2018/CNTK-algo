# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:39:42 2017

@author: tutonso
"""


import os
from PIL import Image
import random
import numpy as np

ClassExisting = ['0','1','2','3','4']
w = 640
l = 772
nChannel = 3
propTest = 0.2


def take_attendance():
    imagesName = [];
#    filePresence = False
    ###list images name of the folder ###
    for item in os.listdir():
        if ".png"  in item:
            imagesName.append(item)
    ###delete in the list, the images name still referenced in "data.txt" ###
#    if os.path.isfile("value.txt"):
#        filePresence = True
    if os.path.isfile("data.txt"):
        #collect data from data.txt
        savedData = open("data.txt",'r')
        dataText = savedData.read().split("\n")
        savedData.close()
        #delete the name in the list of image name of the folder
        for index in range(len(dataText)):
            try :
                name = dataText[index].split("\t")[0].split("\\")[-1]
                if name != "":
                    imagesName.remove(name)
            except ValueError : #if the image still referenced is not in the folder, an error is raised
                raise BaseException("The image " + name + " is referenced but doesn't exist")
#            print(imagesName)
#        else : #if "value.txt" or "data.txt" exist and not the other one, it raises an error 
#            raise BaseException("Images are not referenced but not data used to computed the mean of images exist.\nCould you remove value.txt and reload this script ?")
#    else:
#        if os.path.isfile("data.txt"):
#            raise BaseException("Images are referenced but not there data used to computed the mean.\nCould you remove data.txt and reload this script ?")
    return(imagesName)


def classification():
    imagesName = take_attendance()
    ###Communication with the user :  explanation of waht happens ###
    if len(imagesName)==0:
        print("il n'y a pas d'image à classer ! Rajoutez des images dans le dossier si vous voulez travailler!")
    else:
        print("There is {0} images to classify, be courageous and patient ! \n".format(len(imagesName)))    
        print("Chaque image apparaitra à votre écran.\nPour classifier l'image refermez là, entrez le numéros correspondant à sa classe dans la console, puis appuyer sur Entrer")
        print("Vous pouvez arretez à tous moment en répondant STOP" )
    
        ###Open the files###
        savedData = open("data.txt",'a')  
           
        ###Openning of each image and wait an answer to classify###
        ###if the answer is not correct the image is passed###
        for name in imagesName:
            print("\nFermez l'image puis, \nEntrez 0 s'il s'agit d'une berge ponton,\nEntrez 1 si il s'agit d'une berge forestière, \nEntrez 2 si il s'agit d'une berge-plage, \nEntrez 3 s'il s'agit d'une berge rocailleuse, \nEntrez 4 s'il s'agit d'une berge habitée, \nEntrez STOP si vous voulez arreter la classification, \nEntrez PASSE si vous voulez passer cette image")
            Image.open(name).show()
            answer = input("Votre réponse est :  ")
            if answer == 'PASSE':
                pass
            else:
                if answer == 'STOP':
                    savedData.close()
                    print("Fin de la classification")
                    break
                elif answer not in ClassExisting:
                    print("La réponse que vous venez de donner n'est pas valide, ce n'est pas grave, passons à l'image suivante. Cette image sera reproposé plutard.")
                else:
                    savedData.write(os.path.abspath(name)+"\t"+answer +"\n")
        savedData.close()
    
def sortImages():
    ###Get the image name and the class associated ###
    savedData = open("data.txt",'r')
    data = savedData.read().split("\n")  

    for index in range(len(data)):
        data[index] = data[index].split("\t")
#    data = data[0:-1]
    ###Compute the proportion of image in each class ###
    nbrTest = len(data)*propTest  # number of image need for tests
    ClassProp = [0]*len(ClassExisting)  #proportion of image in each class
    for elt in data:
        ClassProp[int(elt[1])] += 1
    for index in range(len(ClassProp)):
        ClassProp[index] = ClassProp[index]/len(data)
    ###Collect "randomly" images of each classe for the tests ###
    random.shuffle(data)
    print(data)
    trainImages = [] #List of images (name and class) for test, data will be images for train
    NumClass = [] #List of images class
    NameImg = [] #List of images name
    for index in range(len(data)):
        NameImg.append(data[index][0])
        NumClass.append(data[index][1])
    for index in range(len(ClassExisting)): #Collect images wanted for the training, and delete it in data
        nbImagesPut = 0
        while int(ClassProp[index]*nbrTest) > nbImagesPut:
             ind = NumClass.index(ClassExisting[index])
             trainImages.append(data[ind]) 
             del data[ind]
             del NameImg[ind]
             del NumClass[ind]
             nbImagesPut += 1
    moveImages(data,trainImages)
            
def moveImages(testImages,trainImages):
    ### Creation of the folders ###
    if os.path.isdir("train"):
        raise BaseException("The train folder still exists. Could you remove it please ?")
    if os.path.isdir("test"):
        raise BaseException("The test folder still exists. Could you remove it please ?")
    os.mkdir("train")
    os.mkdir("test")
    ###Work on train folder ###
    trainFile = open("train/train.txt",'w')
    name = list(trainImages[0][0])
    name.reverse()
    
    for img in trainImages:
        name = list(img[0])
        name.reverse()
        other = name[name.index('\\'):]
        name = name[:name.index('\\')]
        other.reverse()
        name.reverse()
        name = "".join(name)
        other = "".join(other)
        newName = other + "train\\" + name
        print(newName)
        trainFile.write(other + "train" + name + "\t" + img[1]+"\n")
        os.renames(img[0],newName)
    ###Work on test folder ###
    testFile = open("test/test.txt",'w')
    name = list(testImages[0][0])
    name.reverse()
    for img in testImages:
        name = list(img[0])
        name.reverse()
        other = name[name.index('\\'):]
        name = name[:name.index('\\')]
        other.reverse()
        name.reverse()
        name = "".join(name)
        other = "".join(other)
        newName = other + "test\\" + name
        print(newName)
        testFile.write(other + "test" + name + "\t" + img[1]+"\n")
        os.renames(img[0],newName)
    
    
#classification()
#sortImages()