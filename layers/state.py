from layers.worker_layers import *
from tensorflow.keras import layers as tfkl
from param import *
import numpy as np

class State:
  layers = {}
 
  # misc
  layers['KL'] = tf.Variable(0,trainable = False,dtype = 'float32')
  layers['zeros'] = tf.Variable(np.zeros((BS*BUCKETS,2,2,1)),trainable = False,dtype = 'float32')

  # backbone
  layers['z0_expand_to_z1_expand'],layers['z1_expand_to_z2_expand'],layers['z2_expand_to_z3_expand'],layers['z3_expand_to_z4_expand'],layers['z4_expand_to_z5_expand'] = get_backbone()

  # backbone
  layers['g0_expand_to_g1_expand'],layers['g1_expand_to_g2_expand'],layers['g2_expand_to_g3_expand'],layers['g3_expand_to_g4_expand'],layers['g4_expand_to_g5_expand'] = get_backbone()
  
  # z crush layers
  layers['z1_expand_to_z1'] = FilterCrush(32)
  layers['z2_expand_to_z2'] = FilterCrush(64)
  layers['z3_expand_to_z3'] = FilterCrush(128)
  layers['z4_expand_to_z4'] = FilterCrush(256)
  layers['z5_expand_to_z5'] = FilterCrush(512)

  # z expand layers
  layers['z1_to_z1_expand'] = FilterExpand(32)
  layers['z2_to_z2_expand'] = FilterExpand(64)
  layers['z3_to_z3_expand'] = FilterExpand(128)
  layers['z4_to_z4_expand'] = FilterExpand(256)
  layers['z5_to_z5_expand'] = FilterExpand(512)

  # z_n_expand to z_n-1 infer
  layers['z2_expand_to_z1_expand'] = Infer(32)
  layers['z3_expand_to_z2_expand'] = Infer(64)
  layers['z4_expand_to_z3_expand'] = Infer(128)
  layers['z5_expand_to_z4_expand'] = Infer(256)
  
  # g crush layers
  layers['g1_expand_to_g1'] = FilterCrush(32)
  layers['g2_expand_to_g2'] = FilterCrush(64)
  layers['g3_expand_to_g3'] = FilterCrush(128)
  layers['g4_expand_to_g4'] = FilterCrush(256)
  layers['g5_expand_to_g5'] = FilterCrush(512)

  # g expand layers
  layers['g1_to_g1_expand'] = FilterExpand(32)
  layers['g2_to_g2_expand'] = FilterExpand(64)
  layers['g3_to_g3_expand'] = FilterExpand(128)
  layers['g4_to_g4_expand'] = FilterExpand(256)
  layers['g5_to_g5_expand'] = FilterExpand(512)

  # g_n_expand to g_n-1 infer
  layers['g2_expand_to_g1_expand'] = Infer(32)
  layers['g3_expand_to_g2_expand'] = Infer(64)
  layers['g4_expand_to_g3_expand'] = Infer(128)
  layers['g5_expand_to_g4_expand'] = Infer(256)

  # decoder
  layers['decoder'] = Decoder()
  
  # visual
  layers['alpha_visual'] = Visual()
  layers['beta_visual'] = Visual()
 
  # x distribution layers
  layers['visual_to_x_dist_param_alpha'] = tfkl.Conv2D(1,7,padding = 'same')
  layers['visual_to_x_dist_param_beta'] = tfkl.Conv2D(1,7,padding = 'same')

  # time inference layers
  layers['z5n-1_expand_to_z5n_expand'] = TimeStep(None,512,2)
  layers['z4n-1_expand_to_z4n_expand'] = TimeStep(layers['z5_expand_to_z4_expand'],256,4)
  layers['z3n-1_expand_to_z3n_expand'] = TimeStep(layers['z4_expand_to_z3_expand'],128,8)
  layers['z2n-1_expand_to_z2n_expand'] = TimeStep(layers['z3_expand_to_z2_expand'],64,16)
  layers['z1n-1_expand_to_z1n_expand'] = TimeStep(layers['z2_expand_to_z1_expand'],32,32)
