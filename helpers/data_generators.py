from param import SHAPE,RGB

import glob
import albumentations as A
from PIL import Image
import random 
import os
import math
import numpy as np
import cv2

def get_training_augmentation(max_dim,crop = True,flip = False,light = False):
    train_transform = [A.PadIfNeeded(min_height=max_dim,min_width=max_dim,border_mode = 0)]
    
    if crop:
        train_transform.append(A.OneOf([
            A.RandomSizedCrop(min_max_height=(min(max_dim,256),min(max_dim,256)),height=SHAPE,width=SHAPE,p=1),
            A.RandomSizedCrop(min_max_height=(max_dim,max_dim),height=SHAPE,width=SHAPE,p=1)]
            ,p=1))
    else:
        train_transform.append(
                A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1))

    if flip:
        train_transform.append(A.VerticalFlip(p=.5)) 
        train_transform.append(A.RandomRotate90(p=.5)) 
    if light:
        train_transform.append(A.CLAHE(p=0.8)) 
        train_transform.append(A.RandomBrightnessContrast(p=0.8)) 
        train_transform.append(A.RandomGamma(p=0.8)) 
    return A.Compose(train_transform)


def get_validation_augmentation(max_dim):
    test_transform = [
        A.PadIfNeeded(min_height=max_dim,min_width=max_dim,border_mode = 0),
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(test_transform)


def img_gen(batch_size,path,shape = (SHAPE,SHAPE),crop = True,flip = False,light=False):
  
    image_path_list = []
    for p in path: 
        path_list =list(glob.iglob(p+'/**/*.jpg',recursive = True))
        image_path_list += path_list
  
    X_s = np.zeros((batch_size, shape[1], shape[0],RGB), dtype='float32')

    def getitem(i):
        n = 0

        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open(x))
            max_dim = max(image.shape[0],image.shape[1]) 
            
            aug = get_training_augmentation(max_dim,crop = crop,flip = flip,light = light
                    )(image=image)
            image = aug['image']
            X_s[n] = image[...,np.newaxis]
            n = n + 1
        return X_s  
      
    def on_epoch_end():
        random.shuffle(image_path_list)

    i = -1
    while True :
        if i < len(image_path_list) // batch_size:
            i = i + 1
        else:
            on_epoch_end()
            i = 0
        yield getitem(i)

def tr_gen(batch_size,tn_path,shape = (SHAPE,SHAPE),
        crop = False,flip = False,light = False):
    
    gens = []
    for path in tn_path:
        tn_gen = img_gen(batch_size,path,shape,crop,flip,light)
        gens.append(tn_gen)
    
    Tn = []
    for _ in tn_path:
        T = np.zeros((batch_size, shape[1], shape[0],RGB), dtype='float32') 
        Tn.append(T)            
 
    while True :
        tn = []
        for i in range(len(tn_path)):
            t = next(gens[i])
            tn.append(t)
      
        for i in range(len(tn_path)):
          Tn[i][:,:,:,:] = tn[i][:,:,:,:]

        yield (Tn,{})


