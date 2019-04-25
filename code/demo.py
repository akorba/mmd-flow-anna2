import torch 

from Datasets import *
import os
from torch.autograd import Variable

from model_generator import *

import numpy as np
import torch.optim as optim
#from kcrf.estimator import simple_estimator as np_simple_est
from Utils import * 
import time 
#import matplotlib.pyplot as plt
#%matplotlib inline

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
use_cuda = True
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1




seed = 12
noise_std = 0.0
dname = "ring"
target = load_data(dname, D=2, valid_thresh=0.0, noise_std = noise_std, seed=seed, itanh=False, whiten=True)
D = target.D
target = target_wrapper(target, dtype = torch.float32, device = device)
H = 4
d_out = 1


# Create the generator
center  = 3.
affine_transform = np.random.rand(2,2)

netG = EllipseGenerator(target,center,D).to(device)
torch.manual_seed(999)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
	netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
#netG.apply(weights_init)

# Print the model
#print(netG)

# base_distribution = torch.distributions.multivariate_normal.MultivariateNormal(-2.*torch.ones(2), torch.eye(2))
# netG = EllipseGenerator(target,center,D).to(device)
# torch.manual_seed(999)
# Loss_SMMD = ScaledMMD(D, H, d_out,use_cuda).to(device)
# torch.manual_seed(999)
# Loss = Loss_SMMD
# lr = 0.0002
# optimizerD = optim.Adam(Loss.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# init_data = witness_wrapper(Loss,target,netG)
# G_losses, D_losses = train(Loss,optimizerG, optimizerD,netG, target ,base_distribution,b_size=500)
# final_data = witness_wrapper(Loss,target,netG)
# plot_witness_loss(D_losses,init_data,final_data,  net="D", method = "", num_final_it = 1000)


torch.manual_seed(999)
num_particles = 1000
# Target
D = 2
center_target = 0.
sigma_target = 1.
target = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(D ,dtype=torch.float32,device=device), torch.eye(D,dtype=torch.float32,device=device))
target = target_wrapper(target, dtype = torch.float32, device = device)
#target = GaussianGenerator(target,center_target,sigma_target,D).to(device)
#target = ParticleGenerator(target,num_particles)


# Initializing particles
center = 10.
sigma = .5

base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(D ,dtype=torch.float32,device=device), torch.eye(D,dtype=torch.float32,device=device))
base = target_wrapper(base, dtype = torch.float32, device = device)
base = GaussianGenerator(base,center,sigma,D).to(device)
particles = ParticleGenerator(base,num_particles)

# Building loss
H = 4
d_out = 1
Loss = ScaledMMD( D,H, d_out,use_cuda).to(device)
torch.manual_seed(999)

lr = 1.*num_particles
lr_critic = 0.0002
optimizerD = optim.Adam(Loss.parameters(), lr=lr_critic, betas=(beta1, 0.999))
optimizerG = optim.SGD(particles.parameters(), lr=lr)
init_data = witness_wrapper(Loss,target,particles)
out = train(Loss,optimizerG, optimizerD,particles, target, base_distribution = target , device=device,generator_steps = 100,learn_critic= True,b_size = 1000,save_particles=True)
final_data = witness_wrapper(Loss,target,particles)











