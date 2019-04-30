import torch as tr
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
from PIL import Image

FDTYPE = tr.float32
DEVICE = 'cuda'
#dtype = tr.FloatTensor
#dtype = tr.cuda.DoubleTensor

#from settings import FDTYPE

# def construct_index(dim,s="o", n=1):
# 	''' construct string for use in tf.einsum as it does not support...'''
# 	s = ord(s)
# 	return ''.join([str(unichr(i+s)) for i in range(len(dim)*n)])

# c = 30
# '''
# nl   = lambda x: tf.log(1+tf.exp(x))
# dnl  = lambda x: 1/(1+tf.exp(-x))
# d2nl = lambda x: tf.exp(-x)/tf.square(1+tf.exp(-x))

# '''
# nl   = lambda x: tf.where(x<c, tf.log(1+tf.exp(x)), x)
# dnl  = lambda x: tf.where(x<-c, tf.zeros_like(x), 1/(1+tf.exp(-x)))
# d2nl = lambda x: tf.where(tf.logical_and(-c<x, x<c), tf.exp(-x)/tf.square(1+tf.exp(-x)), tf.zeros_like(x))

# nl   = lambda x: tf.where(x<0, tf.zeros_like(x), 0.5*tf.square(x))
# dnl  = lambda x: tf.where(x<0, tf.zeros_like(x), x)
# d2nl = lambda x: tf.where(x<0, tf.zeros_like(x), tf.ones_like(x))

def pow_10(x): 

	return tr.pow(tr.tensor(10., dtype=FDTYPE, device = DEVICE),x)

# @tf.custom_gradient
# def fix_point_id(var):
# 	beta,S,K,lamda = var

# 	def grad(dvar):
# 		dbeta,dS,dK,dlamda = dvar
# 		npoint = tf.shape(self.X_base)[0]
# 		quad = ( tf.multiply(S,K)-tf.einsum('i,kj,j->ik',S,K_XX,S)+

# 			tf.eye(npoint, dtype=FDTYPE)*lamda
# 			)
# 		inv_quad = tf.linalg.inv(quad)
# 		return -inv_quad*dS
# 	return beta, grad


# def fix_point_id(var):
# 	beta, F, Jac = var

# 	@tf.custom_gradient
# 	def _fix_point_id(F):
# 		def grad(dvar):

# 			return - tf.matrix_solve(Jac, F[:,None])[:,0]
# 		return beta, grad
# 	return _fix_point_id(F)

def support_1d(fun, x):
    assert 1<=x.ndim<=2
    return fun(x) if x.ndim == 2 else fun(x[None,:])[0]

def get_grid(r, i, j, cond):


    grid = np.meshgrid(r,r)

    grid = np.stack(grid,2)
    grid = grid.reshape(-1,2)
    
    num_point = len(grid)
    grid_cond = np.tile(cond[None,:], [num_point, 1])
    
    grid_cond[:,i] = grid[:,0]
    grid_cond[:,j] = grid[:,1]
    return grid_cond
def make_grid_points(D,ngrid,lim):
    idx_i, idx_j = 0,1
    eval_grid = np.linspace(-lim,lim,ngrid)
    cond_values = np.zeros(D)
    epsilon = 1.5
    eval_points = get_grid(eval_grid, idx_i, idx_j, cond_values)
    return eval_points, eval_grid


def load(fn='', size=200, max_samples=None):
    # returns x,y of black pixels
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv,x = np.nonzero(pic)
    y = size-y_inv-1
    if max_samples and x.size > max_samples:
        ixsel = np.random.choice(x.size, max_samples, replace=False)
        x, y = x[ixsel], y[ixsel]
    return x,y




