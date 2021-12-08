# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:17:48 2021

@author: Galanton Andrei-Constantin

"""

#biblioteci pentru transformare imagini in numPy Array

from PIL import Image as PILImage
import os
import numpy as np
import random
import matplotlib.pyplot as plt


def main():
    
    '''
        Partea 1: Transformare set de imagini in NumPy Array
    '''
    
    #initializare NumPy array
    
    numPyArray = []

    
    #initializare path pt folderul cu imagini
    #folderul trebuie sa contina doar imagini si nu subfoldere, etc
    
    pathToImages = "E:\Andrei\PEX - NTT\Cod\PEX-5\procesareDeImagini\imgDataset2"
    
    #parcurgere continut folder (path-ul fiecarei imagini) si transformare imagini in numPy Array
    for imageName in os.listdir(pathToImages):
        
        imageAbsolutePath = os.path.join(pathToImages, imageName)
        
        #print(imageAbsolutePath)        
        
        image = PILImage.open(imageAbsolutePath)        
        imageArray = np.array(image)
        
        numPyArray.append(imageArray)
        
        
    finalNumpyArray = np.array(numPyArray)
    
    print(finalNumpyArray)
    
    #salvare rezultat in fisier NPY pentru algoritmii care cer ca input fisier NPY
    #a se schimba cu Path-ul dorit
    np.save('imgDatasetProcessed.npy',finalNumpyArray,allow_pickle=True)
    
    
    #exemplu Load din fisier .npy   
    
    data1 = np.load('imgDatasetProcessed.npy',allow_pickle=True)
    print(type(data1[0][0][0][0]))  

    
    #exmplu printare imagine random din numPyArray
    
    random_image = random.randint(0, len(finalNumpyArray))
    plt.imshow(finalNumpyArray[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()
    
     
     
        
if __name__ == '__main__':
    main()
        