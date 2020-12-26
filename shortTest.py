#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#tf.config.optimizer.set_jit(True)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[2]:


from layers.filler import myModel
from helpers.data_generators import *
from param import MC,BS,SHAPE,RGB
from tabulate import tabulate

import gc
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import math
import pandas as pd
from PIL import Image
import numpy as np
import scipy.stats
import os
import random
from tensorflow.keras.losses import MAE as mae
from tensorflow.keras.losses import binary_crossentropy as bce
from tensorflow.keras.backend import flatten 
from layers.state import State
l = State.layers
from param import *

MODEL_PATH = './'
WEIGHT_PATH = MODEL_PATH + 'weights.tf'

t0_path = ['/home/judah/Desktop/Cleaned_Bucket_Flagship/healthy']
t1_path = ['/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day0',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day1',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day2',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day3',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day4']
t2_path = ['/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day5',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day6',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day7',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day8',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day9']
t3_path = ['/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day10',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day11',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day12',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day13',
            '/home/judah/Desktop/Cleaned_Bucket_Flagship/ncp/day14']

tn_path = [t0_path,t1_path,t2_path,t3_path]

gen = tr_gen(batch_size = BS,tn_path=tn_path)
model = myModel()
model_opt = tf.keras.optimizers.Adam(lr=.0003,clipnorm = 1.,clipvalue = 0.5)
model.compile(model_opt)

def save_latent_image(z,path):
        
        # scale latent image
        z = z[:,:,0]
        z = z + z.min()
        z = z/z.max()
        z = z*255
        z = z.astype('uint8')
        
        # convert it Image type
        z_im = Image.fromarray(z,mode = 'L')

        # resize
        z_im = z_im.resize((512,512))
        
        # save
        z_im.save(path)

def save_image(x,path):
        x = x.astype('uint8')
        #
        x_im = Image.fromarray(x,mode = 'L')
        #
        x_im = x_im.resize((512,512))
        #
        x_im.save(path)
    
def printStuff(stage,gt,x,x_recon,i):
        # get latents
        z1 = x_recon[stage+'_z1_sample'][gt*BS].numpy()
        z2 = x_recon[stage+'_z2_sample'][gt*BS].numpy()
        z3 = x_recon[stage+'_z3_sample'][gt*BS].numpy()
        z4 = x_recon[stage+'_z4_sample'][gt*BS].numpy()
        z5 = x_recon[stage+'_z5_sample'][gt*BS].numpy()
        # get images
        t0 = x_recon[stage+'_x_reconstructed'][gt*BS].numpy()
        t0 = t0[:,:,0]*255
        x = x[gt][0,:,:,0]
        # save all
        save_image(x,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_x.jpg')
        save_image(t0,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'.jpg')
        save_latent_image(z1,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_z1.jpg')
        save_latent_image(z2,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_z2.jpg')
        save_latent_image(z3,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_z3.jpg')
        save_latent_image(z4,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_z4.jpg')
        save_latent_image(z5,MODEL_PATH + 'pred/t'+str(gt)+'/'+stage+'_'+str(i)+'_z5.jpg')

def predict(g,num,training = True,gen = False):
        for i in range(num):
                x = next(g)[0]
                # 
                x_recon = model(x,training = training,gen = gen)
                # 
                for other_time in range(BUCKETS):
                  for time in range(BUCKETS):
                    printStuff('t{}'.format(time),other_time,x,x_recon,i)    
              
def setKL(kl):
  tf.keras.backend.set_value(l['KL'], kl)


def klSchedule(epoch):
    if epoch == 0:
      kl = 1e-7
    elif epoch < 100:
      kl_per_step = .1/100.0
      kl = kl_per_step*epoch
    elif epoch < 200:
      kl_per_step = .9/100.0
      kl = .1 + kl_per_step*(epoch-100)
    else:
      kl = 1.0
    
    return kl



model.save_weights('./fresh_weights/weights.tf')
for i in range(400):
  kl = klSchedule(i)
  setKL(kl)
  model.fit(gen,epochs = 1,steps_per_epoch = 100)
  if (i % 20) == 0:
    model.save_weights('./weights.tf')
