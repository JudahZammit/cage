from layers.gaussian_dist import * 
from layers.x_dist import *
from layers.state import State
from param import *
l = State.layers
import tensorflow.keras as tfk
import numpy as np

def time_slice(x,t):
    if isinstance(x,list):
      new_x = []
      for j in x:
        if isinstance(j,tuple):
          x1 = j[0][t*BS:(t+1)*BS]
          x2 = j[1][t*BS:(t+1)*BS]
          new_x.append((x1,x2))
        else:
          new_x.append(j[t*BS:(t+1)*BS])
    else:
      if isinstance(x,tuple):
        x1 = x[0][t*BS:(t+1)*BS]
        x2 = x[1][t*BS:(t+1)*BS]
        new_x = (x1,x2)
      else:
        new_x = x[t*BS:(t+1)*BS]

    return new_x

def get_qhat_znt_expand_direct(zn_1_expand,n,var):
  
  zn_expand = l['{}{}_expand_to_{}{}_expand'.format(var,n-1,var,n)](zn_1_expand)

  return zn_expand


def get_qhat_zt_expand_direct(x,var = 'z'):
    
  qhat_zt_expand_direct = {}
   
  zn_1t_expand = x
  for n in range(1,6):
      zn_1t_expand = get_qhat_znt_expand_direct(zn_1t_expand,n,var)
      qhat_zt_expand_direct['qhat_{}t_expand_direct{}'.format(var,n)] = zn_1t_expand

  return qhat_zt_expand_direct

def get_qhat_zn_expand_znt_expand(znt_expand,previous_expands,n):

  t_variable = np.zeros((BS*BUCKETS))
  
  zn_expand = [znt_expand]
  for time in range(BUCKETS):
    if n == 5:
      zn_expand.append(l['z{}n-1_expand_to_z{}n_expand'.format(n,n)]((zn_expand[-1]),t_variable))
    else:   
      zn_expand.append(l['z{}n-1_expand_to_z{}n_expand'.format(n,n)]((zn_expand[-1],previous_expands[time]),t_variable))
    t_variable = (t_variable+1)%BUCKETS
  
  # use the cycled value as the zn_expand
  zn_expand_cycle = zn_expand[-1]
  zn_expand = zn_expand[:-1]
  #zn_expand[0] = zn_expand_cycle

  return zn_expand

def get_all_q_expands(x):
  
  direct_expands = get_qhat_zt_expand_direct(x)

  previous_expands = None
  all_expands = []
  for n in range(5):
    effective_n = 5-n
    previous_expands = get_qhat_zn_expand_znt_expand(
                                  direct_expands['qhat_zt_expand_direct{}'.format(effective_n)],
                                  previous_expands,
                                  effective_n)

    all_expands.append(previous_expands)

  return all_expands

def reorder_layer(layer):
  
  new_layers = []
  for t in range(BUCKETS):
    new_layers.append([])

  for n in range(BUCKETS):
    for t in range(BUCKETS):
      slice_ = time_slice(layer[t],n)
      new_layers[(t+n)%BUCKETS].append(slice_)    

  for t in range(BUCKETS):
    new_layers[t] = tf.concat(new_layers[t],axis = 0)
  
  return new_layers

def reorder_all_expands(all_expands):

  new_all_expands = []
  for layer in all_expands:
    new_all_expands.append(reorder_layer(layer))
    
  
  return new_all_expands

def get_all_qhat_z(all_expands):
  
  qhat_z = []

  for i in range(len(all_expands)):
    qhat_z_layer = []
    for expand in all_expands[i]:
      qhat_z_layer.append(l['z{}_expand_to_z{}'.format(5-i,5-i)](expand))
    qhat_z.append(qhat_z_layer)

  return qhat_z

def get_all_qhat_g(all_expands):
  
  qhat_g = []

  for i in range(len(all_expands)):
    qhat_g_layer = l['g{}_expand_to_g{}'.format(5-i,5-i)](all_expands['qhat_gt_expand_direct{}'.format(5-i)])
    qhat_g.append([qhat_g_layer])

  return qhat_g

def get_qhat_z(x):
  
  all_expands = get_all_q_expands(x)

  reorderd_expands = reorder_all_expands(all_expands)
  
  qhat_z = get_all_qhat_z(reorderd_expands) 
    
  return qhat_z

def get_qhat_g(x):
  
  g_expand = get_qhat_zt_expand_direct(x,'g')

  qhat_g = get_all_qhat_g(g_expand) 
    
  return qhat_g


def combine_params(q_mean,q_logvar,p_mean,p_logvar):
  
  q_var = K.exp(q_logvar)
  q_var_inv = 1/q_var

  p_var = K.exp(p_logvar)
  p_var_inv = 1/p_var

  var = 1/(p_var_inv + q_var_inv)
  logvar = K.log(var)

  mean_numerator = q_mean*q_var_inv + p_mean*p_var_inv
  mean_denominator = (p_var_inv + q_var_inv)
        
  mean = mean_numerator/mean_denominator

  return mean,logvar

def get_level_info(qhat_zn_1,zn_expanded,zn_sample,level,gen,var):
    
  if level == 5:
    p_zn_1 = get_unit_gaussian_dist()
  else:
    zn_1_expanded = l['{}{}_expand_to_{}{}_expand'.format(var,level+1,var,level)](zn_expanded)
    p_zn_1 = l['{}{}_expand_to_{}{}'.format(var,level,var,level)](zn_1_expanded)   

  
  if gen:
    q_zn_1 = p_zn_1
  else:
    q_zn_1 = combine_params(qhat_zn_1[0],qhat_zn_1[1],
                            p_zn_1[0],p_zn_1[1])
    
  zn_1_sample = gaussian_sample(q_zn_1[0],q_zn_1[1])
  zn_1_expanded = l['{}{}_to_{}{}_expand'.format(var,level,var,level)](zn_1_sample)    
   

  return p_zn_1,q_zn_1,zn_1_expanded,zn_1_sample


def time_zero_information(qhat_z,var = 'z',gen = False):
  
  qhat_z = [qhat_z[0][0],
            qhat_z[1][0],
            qhat_z[2][0],
            qhat_z[3][0],
            qhat_z[4][0]]
  p_z = []
  q_z = []
  z_expanded = []
  z_sample = []

  level_5 = get_level_info(qhat_z[0],None,None,5-0,gen,var)
  p_z.append(level_5[0])
  q_z.append(level_5[1])
  z_expanded.append(level_5[2])
  z_sample.append(level_5[3])
  for i in range(1,5):
    level = 5 - i
    level_n_1 = get_level_info(qhat_z[i],z_expanded[-1],z_sample[-1],level,gen,var)
    p_z.append(level_n_1[0])
    q_z.append(level_n_1[1])
    z_expanded.append(level_n_1[2])
    z_sample.append(level_n_1[3])

  out={}
  out['p_z'] = p_z
  out['q_z'] = q_z
  out['z_expanded'] = z_expanded
  out['z_sample'] = z_sample

  return out

def infected_level_info(lower_z_expanded,right_z_expanded,qhat_z,level,time,gen):
    
  t_variable = np.zeros((BS*BUCKETS))  
  t_variable = (t_variable+(time-1))%BUCKETS
  
  if level == 5:
    z_expanded = l['z{}n-1_expand_to_z{}n_expand'.format(level,level)]((right_z_expanded),t_variable) 
  else:
    z_expanded = l['z{}n-1_expand_to_z{}n_expand'.format(level,level)]((right_z_expanded,lower_z_expanded),t_variable) 

  
  p_z = l['z{}_expand_to_z{}'.format(level,level)](z_expanded)
  
  if gen:
    q_z = p_z
  else:
    q_z = combine_params(qhat_z[0],qhat_z[1],
                            p_z[0],p_z[1])

  z_sample = gaussian_sample(q_z[0],q_z[1])
  z_expanded = l['z{}_to_z{}_expand'.format(level,level)](z_sample)    

  return p_z,q_z,z_expanded,z_sample

def time_n_to_time_n_plus_1(inputs,qhat_z,time,gen):

  qhat_z = [qhat_z[0][time],
            qhat_z[1][time],
            qhat_z[2][time],
            qhat_z[3][time],
            qhat_z[4][time]]
  
  p_z = inputs['p_z']
  q_z = inputs['q_z']
  z_expanded = inputs['z_expanded']
  z_sample = inputs['z_sample']
  
  p_z_1 = []
  q_z_1 = []
  z_1_expanded = []
  z_1_sample = []
  
  level_5 = infected_level_info(None,z_expanded[0],qhat_z[0],5,time,gen)
  p_z_1.append(level_5[0])
  q_z_1.append(level_5[1])
  z_1_expanded.append(level_5[2])
  z_1_sample.append(level_5[3])
  for i in range(1,5):
    level = 5-i
    level_n_1 = infected_level_info(z_1_expanded[-1],z_expanded[i],qhat_z[i],level,time,gen)
    p_z_1.append(level_n_1[0])
    q_z_1.append(level_n_1[1])
    z_1_expanded.append(level_n_1[2])
    z_1_sample.append(level_n_1[3])
 
  out={}
  out['p_z'] = p_z_1
  out['q_z'] = q_z_1
  out['z_expanded'] = z_1_expanded
  out['z_sample'] = z_1_sample
  
  return out

def get_decoded_z(z_expanded):
    return l['decoder'](z_expanded)

def get_visuals(decoded_z):
    
    alpha_visual = l['alpha_visual'](decoded_z)
    beta_visual = l['beta_visual'](decoded_z)
    
    return alpha_visual,beta_visual

def create_output_dict(z_sample,x_reconstructed,t):
  
  out = {}
  out['t{}_z5_sample'.format(t)] = z_sample[0]
  out['t{}_z4_sample'.format(t)] = z_sample[1]
  out['t{}_z3_sample'.format(t)] = z_sample[2]
  out['t{}_z2_sample'.format(t)] = z_sample[3]
  out['t{}_z1_sample'.format(t)] = z_sample[4]
  out['t{}_x_reconstructed'.format(t)] = x_reconstructed

  return out

def create_loss_dict(xent,z_sample,p_z,q_z):
  
  # get losses
  loss_dict = {}

  # x recon loss
  loss_dict['XENT'] = xent  

  # p_z loss 
  for i in range(5):
    loss_dict['p_z{}'.format(5-i)] = -gaussian_ll(z_sample[i],p_z[i][0],p_z[i][1]) 
  
  # q_z loss 
  for i in range(5):
    loss_dict['q_z{}'.format(5-i)] = gaussian_ll(z_sample[i],q_z[i][0],q_z[i][1])
  
  loss = 0
  for x in loss_dict.values():
    loss += x
  
  loss_dict['loss'] = loss
  loss_dict['KL'] = loss-loss_dict['XENT'] 

  return loss_dict

def g_loss(z_sample,p_z,q_z):
  
  loss_dict = {}

  # p_z loss 
  for i in range(5):
    loss_dict['p_z{}'.format(5-i)] = -gaussian_ll(z_sample[i],p_z[i][0],p_z[i][1]) 
  
  # q_z loss 
  for i in range(5):
    loss_dict['q_z{}'.format(5-i)] = gaussian_ll(z_sample[i],q_z[i][0],q_z[i][1])
  
  loss = 0
  for x in loss_dict.values():
    loss += x
  
  return loss


def predict(inputs,gen):
  t0,t1,t2,t3 = inputs 
  x = tf.concat([t0,t1,t2,t3],axis = 0) 
 
  qhat_z = get_qhat_z(x)
  qhat_g = get_qhat_g(x)

  g_info = time_zero_information(qhat_g,var = 'g',gen = gen)  
  znm_info = []
  znm_info.append(time_zero_information(qhat_z,gen = gen))     
  for bucket in range(1,BUCKETS):
    znm_info.append(time_n_to_time_n_plus_1(znm_info[-1],qhat_z,bucket,gen))     
   
  all_out = {}
  all_loss = {'KL':0,'loss':0,'XENT':0}

  g_expand = g_info['z_expanded']   
 
  all_expand = []
  for info in znm_info:
    all_expand.append(info['z_expanded'])
 
  all_expand_concat = []
  for level in range(5):
    level_expand = []
    for time in range(BUCKETS):
      level_expand.append(all_expand[time][level])
    level_expand = tf.concat(level_expand,axis = 0)
    all_expand_concat.append(level_expand) 
     
  
  all_expand = all_expand_concat  
  for i in range(len(all_expand)):
    rep = tf.tile(g_expand[i],[BUCKETS,1,1,1])
    all_expand[i] = tf.concat([all_expand[i],rep],axis = -1)

  all_decoded = get_decoded_z(all_expand)
  all_alpha_visual,all_beta_visual = get_visuals(all_decoded)
  
  alpha_visual = []
  beta_visual = []
  for time in range(BUCKETS):
      alpha_visual_t = all_alpha_visual[time*BS*BUCKETS:(time+1)*BS*BUCKETS]
      beta_visual_t = all_beta_visual[time*BS*BUCKETS:(time+1)*BS*BUCKETS]
      alpha_visual.append(alpha_visual_t)
      beta_visual.append(beta_visual_t)

  t = 0
  for info in znm_info:
    p_x = visual_to_x_dist(alpha_visual[t],beta_visual[t])
    x_reconstructed = dist_to_x(p_x)
    xent = -x_ll(time_slice(x,t),time_slice(p_x,t))
 
    out = create_output_dict(info['z_sample'],x_reconstructed,t)
    all_out.update(out)
   
    loss_dict = create_loss_dict(xent,info['z_sample'],
                                info['p_z'],info['q_z'])
    for key in all_loss:
        all_loss[key] += loss_dict[key] 
    t += 1

  g_kl_loss = g_loss(g_info['z_sample'],
              g_info['p_z'],g_info['q_z'])

  all_loss['KL'] += g_kl_loss
  all_loss['loss'] += g_kl_loss

  return all_out,all_loss

 
class myModel(tfk.Model):
    def __init__(self):
        super(myModel,self).__init__()
        
        self.l = State.layers

    def call(self,inputs,gen = False):
        out,loss_dict = predict(inputs,gen = gen)
        out['loss'] = loss_dict['loss'] 
        self.add_loss(loss_dict['loss'])
        self.add_metric(loss_dict['XENT'],name = 'XENT',aggregation = 'mean')          
        self.add_metric(loss_dict['KL']/l['KL'],name = 'Actual KL',aggregation = 'mean')       
        self.add_metric(loss_dict['KL'],name = 'Scaled KL',aggregation = 'mean')       

        return out


