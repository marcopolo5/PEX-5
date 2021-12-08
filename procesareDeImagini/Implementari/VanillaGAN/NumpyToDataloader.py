# -*- coding: utf-8 -*-
"""
@author: Galanton Andrei-Constantin
"""



#biblioteci pentru transformare imagini in numPy Array

from PIL import Image as PILImage
import numpy as np
import os
import cv2


#biblioteci pentru transformare numPy Array in Dataset pregatit pt GAN
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random

#posibili sa arate eroare dar sa ruleze corect programul


#blblioteci pentru salvarea Numpy Array ca fisier NPY sau a face load
from numpy import asarray
from numpy import save
from numpy import load


'''
    Creare clasa care sa ajute la crearea Dataset-ului
'''


image_size = 300
DATA_DIR = 'imgDatasetProcessed.npy'
#facadeTrain = np.load(DATA_DIR, allow_pickle=True)

'''
print(f"Shape of training data: {facadeTrain.shape}")
print(f"Data type: {type(facadeTrain)}")


data = facadeTrain.astype(np.uint8)
data = 255 * data
img = data.astype(np.uint8)
facadeTrain = img

print(f"Shape of training data: {facadeTrain.shape}")
print(f"Data type: {type(facadeTrain)}")

'''




class facadeDataset(Dataset):
    
    #initializare clasa cu numPyArray-ul create la Partea 1
    def __init__(self, finalNumpyArray):
        self.finalNumpyArray = finalNumpyArray
        
    #functie care returneaza numarul de elemente din Dataset        
    def __len__(self):        
        return len(self.finalNumpyArray)
    
    #functie care returneaza un item din Dataset
    def __getitem__(self, index):
        
        image =  self.finalNumpyArray[index]
        result = self.transform(image)
        return result
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(size=(image_size,image_size)),
        #T.RandomResizedCrop(image_size),
        #T.RandomHorizontalFlip(),
        T.ToTensor()])
        
    
    
        
    

    #Functii de vizualizare imagini si batchuri din dataset


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
def show_batch(dl, nmax=64):
    for images in dl:
        #print(images)
        show_images(images, nmax)
        break

    
'''
------------------------------------------Main-----------------------------------------
'''

def main():
    
    
    
    '''
        Partea 2: Transformare NumPy Array in Dataset pentru GAN    
    '''
    
    #verificare tip de date numPy Array
    #Trebuie sa fie numpy.uint8 (in caz contrar se face tranformarea)
    '''
    print(type(finalNumpyArray[0][0][0][0]))
    '''
    
    #exemplu transformare tip Array din float64 in uint8
    '''
    data = finalNumpyArray.astype(np.float64)
    data = 255 * data
    finalNumpyArray = data.astype(np.uint8)
    '''
    
    
    
    batch_size = 64
    data = np.load('imgDatasetProcessed.npy',allow_pickle=True)
    transformed_dataset = facadeDataset(data)
    train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    show_batch(train_dl)  
    
    
    

        
if __name__ == '__main__':
    main()
        
        
        