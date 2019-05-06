import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from Datasets import *
from gaussian import *
from Utils import *
import os
import numpy as np


class EllipseGenerator(nn.Module):
	def __init__(self,noise_generator,center,d):
		super(EllipseGenerator,self).__init__()
		self.noise_generator = noise_generator
		self.center = nn.Parameter(tr.tensor(center))
		#self.affine_transform = nn.Parameter(tr.tensor(center))
		self.linear = tr.nn.Linear(d, d, bias=False)
	def forward(self, sample):
		return self.center+ self.linear(sample)
	def sample(self,num_samples):
		noise = self.noise_generator.sample(num_samples)
		out = self.forward(noise)
		return out.detach()

class GaussianGenerator(nn.Module):
	def __init__(self,noise_generator, center,sigma,d):
		super(GaussianGenerator,self).__init__()
		self.noise_generator = noise_generator
		self.center = nn.Parameter(tr.tensor(center))
		self.sigma = nn.Parameter(tr.tensor(sigma))
		self.dtype = noise_generator.dtype
		self.device = noise_generator.device
	def forward(self,sample):
		return self.center + self.sigma*sample
	def sample(self,num_samples):
		noise = self.noise_generator.sample(num_samples)
		out = self.forward(noise)
		return out.detach()

class ParticleGenerator(nn.Module):
	def __init__(self,base,num_particles):
		super(ParticleGenerator,self).__init__()
		self.base = base
		samples = base.sample(num_particles).clone().detach().requires_grad_(True)
		self.particles  = nn.Parameter(samples)
		self.D = self.particles.shape[1]
		self.dtype = base.dtype
		self.device = base.device
	def forward(self,sample):
		return self.particles + 0.*tr.mean(sample)
	def sample(self,num_samples):
		idx = tr.randperm(self.particles.shape[0])		
		return self.particles[idx[:num_samples]].detach()


class ShapeGenerator(nn.Module):

	def __init__(self,filename,num_particles,size = 200, dtype= tr.float32, device = 'cpu'):
		super(ShapeGenerator,self).__init__()
		xP,yP=load(filename, size, num_particles)
		samples = tr.tensor(np.stack((xP,yP),1), dtype = dtype, device = device).float() / size * 2 - 1
		self.particles  = nn.Parameter(samples)
		self.D = self.particles.shape[1]
	def forward(self,sample):
		return self.particles + 0.*tr.mean(sample)
	def sample(self,num_samples):
		idx = tr.randperm(self.particles.shape[0])		
		return self.particles[idx[:num_samples]].detach()	


class TwoLayerCritic(nn.Module):
	def __init__(self,d_int, H, d_out):
		super(TwoLayerCritic,self).__init__()

		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		self.linear1 = tr.nn.Linear(d_int, 4*H)
		self.linear2 = tr.nn.Linear(4*H, 2*H)
		self.linear3 = tr.nn.Linear(2*H, H)
		self.linear4 = tr.nn.Linear(H, d_out)

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		h1_relu = self.linear1(x).clamp(min=0)
		h2_relu = self.linear2(h1_relu).clamp(min=0)
		h3_relu = self.linear3(h2_relu).clamp(min=0)
		h4_relu = self.linear4(h3_relu)
		return h4_relu
class Identity(nn.Module):
	def __init__(self,sigma=1):
		super(Identity,self).__init__()
		self.sigma = sigma
	def forward(self,x):
		return x/self.sigma

class Norm(nn.Module):
	def __init__(self):
		super(Norm,self).__init__()
	def forward(self,x):
		norm =  x.norm(2,dim=1)

		return norm.unsqueeze(-1)
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	# if classname.find('Linear'): 
	# 	nn.init.uniform_(m.weight)
	# 	nn.init.uniform_(m.bias)


# def mmd2(kernel,true_data, fake_data):	

# 	gram_XY = kernel.kernel(true_data,fake_data)
# 	gram_XX = kernel.kernel(true_data, true_data)
# 	gram_YY = kernel.kernel(fake_data,fake_data)
# 	N_x, _ = gram_XX.shape
# 	N_y, _ = gram_YY.shape
# 	mmd2 = (1./(N_x*(N_x-1)))*(tr.sum(gram_XX)-tr.trace(gram_XX)) \
# 		+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY)-tr.trace(gram_YY)) \
# 		- 2.* tr.mean(gram_XY)
# 	return mmd2



class mmd2(tr.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, kernel,true_data,fake_data):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		#noisy_data = fake_data + noise
		#fake_data = fake_data.clone().detach()



		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,fake_data)
			gram_XX = kernel.kernel(true_data, true_data)
			gram_YY = kernel.kernel(fake_data,fake_data)
			N_x, _ = gram_XX.shape
			N_y, _ = gram_YY.shape
			mmd2 = (1./(N_x*(N_x-1)))*(tr.sum(gram_XX)-tr.trace(gram_XX)) \
				+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY)-tr.trace(gram_YY)) \
				- 2.* tr.mean(gram_XY)
			mmd2_for_grad = 0.5*N_y*mmd2
		ctx.save_for_backward(mmd2_for_grad,fake_data)

		return 0.5*mmd2

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		mmd2_for_grad, fake_data = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=fake_data,
					  	grad_outputs=grad_output,
					 	create_graph=True, only_inputs=True)[0] 
				
		return None, None, gradients





class mmd2_smoothed(tr.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, kernel,true_data,fake_data,noisy_data):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		#noisy_data = fake_data + noise
		#fake_data = fake_data.clone().detach()



		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,noisy_data)
			gram_XX = kernel.kernel(true_data, true_data)
			gram_YY = kernel.kernel(fake_data,noisy_data)
			gram_YY_t = kernel.kernel(fake_data,fake_data)
			gram_XY_t = kernel.kernel(true_data,fake_data)
			N_x, _ = gram_XX.shape
			N_y, N_z = gram_YY.shape
			mmd2_for_grad =  N_z*( 1./(N_y*(N_y-1))*(tr.sum(gram_YY)-tr.trace(gram_YY))  - tr.mean(gram_XY))

			mmd2 = (1./(N_x*(N_x-1)))*(tr.sum(gram_XX)-tr.trace(gram_XX)) \
				+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY_t)-tr.trace(gram_YY_t)) \
				- 2.* tr.mean(gram_XY_t)

		ctx.save_for_backward(mmd2_for_grad,mmd2,noisy_data)

		return 0.5*mmd2.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		mmd2_for_grad,mmd2, noisy_data = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_data,
					  	grad_outputs=grad_output,
					 	create_graph=True, only_inputs=True)[0] 
			gradients = gradients*(mmd2>0)
		return None, None, None, gradients


class MMD_smoothed(nn.Module):
	def __init__(self,critic,noise_level,D,d_out,use_cuda,dtype, device):
		super(MMD_smoothed,self).__init__()
		self.kernel = Gaussian(d_out,1., dtype= dtype,device = device)
		self.critic = critic

		noise_sampler = tr.distributions.multivariate_normal.MultivariateNormal(tr.zeros(D ,dtype=dtype,device=device), tr.eye(D,dtype=dtype,device=device))
		self.noise_sampler = target_wrapper(noise_sampler, dtype = dtype, device = device)
		self.noise_level = noise_level
		self.use_cuda = use_cuda
		self.with_base = False
		self.with_mmd  = False
		self.mmd2 =  mmd2_smoothed.apply
	def forward(self, t_data, f_data,with_mmd=False, rescale=True):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data).clone().detach()
		noise  = self.noise_level*self.noise_sampler.sample(f_data.shape[0])
		noisy_data = self.critic(f_data + noise)
		#noisy_data = self.critic(f_data)

		mmd2_val = self.mmd2(self.kernel,true_data,fake_data,noisy_data)
		return mmd2_val

	def witness(self, t_data,f_data,g_data):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data)
		grid_data = self.critic(g_data)
		gram_f = tr.mean(self.kernel.kernel(grid_data, fake_data),1)
		gram_t = tr.mean(self.kernel.kernel(grid_data, true_data),1)
		return gram_f - gram_t

	def norm_grad_witness(self,t_data,f_data,grid_data):
		witness = self.witness(t_data, f_data,grid_data)
		gradients = autograd.grad(outputs=witness, inputs=grid_data,
							  grad_outputs=tr.ones_like(witness),
							  create_graph=True, retain_graph=True, only_inputs=True)[0] 

		return gradients.norm(2,dim=1)




class MMD(nn.Module):
	def __init__(self,critic,d_out,use_cuda,dtype, device):
		super(MMD,self).__init__()
		self.kernel = Gaussian(d_out,1., dtype= dtype,device = device)
		self.critic = critic
		self.use_cuda = use_cuda
		self.with_base = False
		self.with_mmd  = False
		self.mmd2 =  mmd2.apply
		self.noise_level = 0.
	def forward(self, t_data, f_data,with_mmd=False, rescale=True):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data)
		mmd2_val = self.mmd2(self.kernel,true_data,fake_data)
		return mmd2_val

	def witness(self, t_data,f_data,g_data):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data)
		grid_data = self.critic(g_data)
		gram_f = tr.mean(self.kernel.kernel(grid_data, fake_data),1)
		gram_t = tr.mean(self.kernel.kernel(grid_data, true_data),1)
		return gram_f - gram_t

	def norm_grad_witness(self,t_data,f_data,grid_data):
		witness = self.witness(t_data, f_data,grid_data)
		gradients = autograd.grad(outputs=witness, inputs=grid_data,
							  grad_outputs=tr.ones(witness.size()).cuda() if self.use_cuda else tr.ones(
								  witness.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0] 

		return gradients.norm(2,dim=1)




class MMD_particles(nn.Module):
	def __init__(self,particles,critic,d_out,use_cuda):
		super(MMD,self).__init__()
		self.kernel = Gaussian(d_out,1.)
		self.critic = critic
		self.particles = nn.Parameter(particles)
		self.use_cuda = use_cuda
		self.with_base = False
		self.with_mmd  = False
	def forward(self, t_data):
		true_data = self.critic(t_data)
		fake_data = self.critic(self.particles)
		mmd2_val = self.mmd2(self.kernel,true_data,fake_data)
		return mmd2_val
	def witness(self, t_data,g_data):
		true_data = self.critic(t_data)
		fake_data = self.critic(self.particles)
		grid_data = self.critic(g_data)
		gram_f = tr.mean(self.kernel.kernel(grid_data, fake_data),1)
		gram_t = tr.mean(self.kernel.kernel(grid_data, true_data),1)
		return gram_t - gram_f

	def norm_grad_witness(self,t_data,grid_data):
		witness = self.witness(t_data,grid_data)
		gradients = autograd.grad(outputs=witness, inputs=self.particles,
							  grad_outputs=tr.ones(witness.size()).cuda() if self.use_cuda else tr.ones(
								  witness.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0] 

		return gradients.norm(2,dim=1)

	def expected_witness(self,t_data):
		gradients = norm_grad_witness(t_data,self.particles)
		return gradients.mean()



class AdversarialMMD(MMD):
	def __init__(self,d_int,H, d_out,use_cuda):
		critic = TwoLayerCritic(d_int,H,d_out)
		super(AdversarialMMD,self).__init__(critic,d_out,use_cuda)


class ScaledMMD(AdversarialMMD):
	def __init__(self,*args):
		super(ScaledMMD,self).__init__(*args)
		self.with_base = True
		self.with_mmd  = True
	def forward(self, t_data, f_data,b_data,with_mmd = False):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data)
		base_data = self.critic(b_data)
		mmd2_val = self.mmd2(self.kernel,true_data,fake_data)
		
		gradients = autograd.grad(outputs=base_data, inputs=b_data,
							  grad_outputs=tr.ones(base_data.size()).cuda() if self.use_cuda else tr.ones(
								  base_data.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]
		gradients = gradients.view(gradients.size(0), -1)
		scaling = 1.+ tr.mean(gradients.norm(2,dim=1)**2)
		if with_mmd:
			return mmd2_val/scaling, mmd2_val
		else:
			return mmd2_val/scaling
	def eval_mmd(self,t_data,f_data):
		true_data = self.critic(t_data)
		fake_data = self.critic(f_data)
		mmd2_val = self.mmd2(self.kernel,true_data,fake_data)
		return mmd2_val


class PlainMMD(MMD):
	def __init__(self,sigma,*args):
		critic = Identity(sigma)
		super(PlainMMD,self).__init__(critic,*args)


class PlainMMD_smoothed(MMD_smoothed):
	def __init__(self,sigma,*args):
		critic = Identity(sigma)
		super(PlainMMD_smoothed,self).__init__(critic,*args)

class RadialMMD(MMD):
	def __init__(self,*args):
		critic = Norm()
		super(RadialMMD,self).__init__(critic,*args)



def args_loss(Loss, base_distribution, real_data, fake_data):
	b_size = real_data.shape[0]
	if Loss.with_base and base_distribution is not None:
		base_data = base_distribution.sample(b_size)
		base_data.requires_grad=True
		args = real_data, fake_data,base_data
	else:
		args = real_data, fake_data
	return args


class target_wrapper(object):
	def __init__(self, target, dtype, device):
		self.target = target
		self.dtype = dtype
		self.device = device
		if hasattr(target, 'D'):
			self.D = target.D
		else:
			self.D = target.sample([1]).shape[1]
	def sample(self,num):
		num = [num]

		return tr.tensor(self.target.sample(num), dtype=self.dtype,  device=self.device)


def GradientFlow(Loss, optimizer , target,  device="cuda",b_size=1000, generator_steps=1000,num_epochs=1):

	# Training Loop
	# Lists to keep track of progress
	G_losses = []
	D_losses = []

	print("Starting Training Loop...")
	# For each epoch
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for j in range(generator_steps):

			Loss.zero_grad()
			real_data = target.sample([b_size])
			loss = Loss(real_data)
				# Calculate the gradients for this batch
			loss.backward()
			
			# Update G
			optimizer.step()

				# Save Losses for plotting later
			# Loss.zero_grad()
			# real_data = target.sample([b_size])
			# noise = target.sample([b_size])
			# fake_data = netG(noise)			
			losses.append(loss.item())
			#G_losses.append(Loss.eval_mmd(real_data,fake_data))
			#if j %5==0:
			#	print(loss)
	return losses



def train(Loss,optimizerG, optimizerD,netG, target, base_distribution=None, device="cuda",b_size=1000,b_size_critic=100, critic_steps=100,generator_steps=1,num_epochs=1, learn_critic= True,save_particles = True, writer = None):

	# Training Loop
	# Lists to keep track of progress
	G_losses = []
	D_losses = []
	particles = []

	out = {
			"G_losses":[],
			"D_losses":[],
			"particles":[],
			"particles_iter":[],
			"target_particles":target.sample(b_size).cpu().detach().numpy()
		}

	print("Starting Training Loop...")
	total_iters = 0
	# For each epoch
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for j in range(generator_steps):
			if learn_critic:
				for i in range(critic_steps):
					total_iters +=1
					############################
					# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
					###########################
					## Train with all-real batch
					Loss.zero_grad()
					# Format batch
					real_data 	= target.sample(b_size_critic)
					fake_data 	= netG.sample(b_size_critic)
					#fake_data 	= netG(noise)

					# Calculate D's loss on the all-fake batch
					args = args_loss(Loss, base_distribution,real_data, fake_data)
					loss = -Loss(*args)
					
					# Calculate the gradients for this batch
					loss.backward()
					# Update D
					optimizerD.step()

					############################
					# (2) Update G network: maximize log(D(G(z)))
					###########################
								# Output training stats
					#if i % 500 == 0:
					#	print(loss)
					Loss.zero_grad()
					# Format batch
					if Loss.with_mmd:
						real_data 	= target.sample(b_size_critic)
						fake_data 	= netG.sample(b_size_critic)
						args = args_loss(Loss, base_distribution,real_data, fake_data)

						_, loss = Loss(*args,with_mmd =True)
					#	if phase=="train":

					out = save(writer,out,loss,netG,total_iters,'D_step',save_particles=save_particles)
					#D_losses.append(Loss.eval_mmd(real_data,fake_data))

			netG.zero_grad()
			real_data = target.sample(b_size)
			noise = target.sample(b_size)
			fake_data = netG(noise)
			args = args_loss(Loss, base_distribution,real_data, fake_data)
			loss = Loss(*args)
				# Calculate the gradients for this batch
			loss.backward()
			
			# Update G
			optimizerG.step()

			out = save(writer,out,loss,netG,j,'G_step')
			#if np.mod(j+1,5000)==0:
			#	Loss.noise_level = 0.
				# Save Losses for plotting later
			# Loss.zero_grad()
			# real_data = target.sample([b_size])
			# noise = target.sample([b_size])
			# fake_data = netG(noise)


		print('Epoch: '+ str(epoch) + ' | loss: '+ str(loss))
			#print(str(j))

			#G_losses.append(Loss.eval_mmd(real_data,fake_data))
			#if j %5==0:
			#	print(loss)
	return out



def save(writer,out,loss,netG,iters,mode, save_particles=True):
	if mode=='D_step':
		if writer:
			writer.add_scalars('data/D_losses',{"D_losses":loss},iters)
		out["D_losses"].append(loss.item())
	
	elif mode=='G_step':
		if writer:
			writer.add_scalars('data/G_losses',{"G_losses":loss},iters)
		out["G_losses"].append(loss.item())
		if save_particles and np.mod(iters,10000)==0:
			if writer:
				#final_particles = netG.particles.cpu().detach().numpy()
				writer.add_embedding(netG.particles,global_step= iters)
			out["particles"].append(netG.particles.clone().cpu().detach().numpy())
			out["particles_iter"].append(iters)
		if np.mod(iters,1000)==0:
			print('Saving checkpoint ...')
			print('Epoch: '+ str(iters) + ' | loss: '+ str(loss))
			state = {
				'net': netG.state_dict(),
				'loss': loss,
				'iters':iters,
			}
			if not os.path.isdir(writer.log_dir +'/checkpoint'):
				os.mkdir(writer.log_dir + '/checkpoint')
			tr.save(state,writer.log_dir +'/checkpoint/ckpt.iter_'+str(iters))
	return out




		

def witness_wrapper(loss,target, netG, device="cuda", num_grid_points = 100,lim=20):

	b_size = 10000
	D = target.D
	t_data = target.sample(b_size)
	noise = target.sample(b_size)
	f_data = netG(noise)
	grid_data_np, eval_grid =  make_grid_points(D,num_grid_points, lim = lim)
	grid_data = tr.tensor(grid_data_np, dtype=t_data.dtype,device=t_data.device, requires_grad=True)
	
	witness_val = loss.witness(t_data,f_data,grid_data).cpu().detach().numpy()
	norm_grad_witness = loss.norm_grad_witness(t_data,f_data,grid_data).cpu().detach().numpy()
	return witness_val,norm_grad_witness,t_data.cpu().detach().numpy(), f_data.cpu().detach().numpy(),eval_grid
	
def witness_wrapper_1d(loss,target, netG, device="cuda", num_grid_points = 100,lim=20):

	b_size = 10000
	D = target.D
	t_data = target.sample(b_size)
	noise = target.sample(b_size)
	f_data = netG(noise)
	grid_data_np, eval_grid =  make_grid_points(2,num_grid_points, lim = lim)
	grid_data = tr.tensor(eval_grid, dtype=t_data.dtype,device=t_data.device, requires_grad=True)
	grid_data = grid_data.reshape([-1,1])

	witness_val = loss.witness(t_data,f_data,grid_data).cpu().detach().numpy()
	norm_grad_witness = loss.norm_grad_witness(t_data,f_data,grid_data).cpu().detach().numpy()
	return witness_val,norm_grad_witness,t_data.cpu().detach().numpy(), f_data.cpu().detach().numpy(),eval_grid

