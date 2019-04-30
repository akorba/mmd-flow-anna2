
import matplotlib.pyplot as plt
from Utils import *

def make_grid_points(D,ngrid,lim):
    idx_i, idx_j = 0,1
    eval_grid = np.linspace(-lim,lim,ngrid)
    cond_values = np.zeros(D)
    epsilon = 1.5
    eval_points = get_grid(eval_grid, idx_i, idx_j, cond_values)
    return eval_points, eval_grid
def plot_pdf(fig,ax,witness_val,t_data, f_data, eval_grid,v_min=None,v_max=None,plot_data=True,plot_witness=False): 
    idx_i, idx_j = 0,1
    epsilon = 1.5
    ngrid = eval_grid.shape[0]
    if plot_witness:
        if (v_min is not None) and  (v_max is  not None):
            c= ax.pcolor(eval_grid, eval_grid, witness_val.reshape(ngrid, ngrid),vmin=v_min, vmax=v_max)
        else:
            c= ax.pcolor(eval_grid, eval_grid, witness_val.reshape(ngrid, ngrid))
        fig.colorbar(c, ax=ax)
        ax.set_xlim(eval_grid.min(),eval_grid.max())
        ax.set_ylim(eval_grid.min(),eval_grid.max())
    if plot_data:
        ax.scatter(t_data[:1000,idx_i], t_data[:1000,idx_j], 5, color="k", alpha=0.8, vmin=1, marker="x")
        ax.scatter(f_data[:1000,idx_i], f_data[:1000,idx_j], 5, color="r", alpha=0.8, vmin=1, marker="x")

    
def plot_pdf_1d(fig,ax,witness_val,t_data, f_data, eval_grid,v_min=None,v_max=None,plot_data=True, plot_witness=False): 
    idx_i, idx_j = 0,1
    epsilon = 1.5
    ngrid = eval_grid.shape[0]
    #if (v_min is not None) and  (v_max is  not None):
    #    c= ax.pcolor(eval_grid, eval_grid, witness_val.reshape(ngrid, ngrid),vmin=v_min, vmax=v_max)
    #else:
    #    c= ax.pcolor(eval_grid, eval_grid, witness_val.reshape(ngrid, ngrid))
    if plot_witness:
        ax.plot(eval_grid,witness_val)
    if plot_data:
        zeros = np.zeros([len(t_data[:1000,0])])
        ax.scatter(t_data[:1000,idx_i], zeros, 5, color="k", alpha=0.8, vmin=1, marker="x")
        ax.scatter(f_data[:1000,idx_i], np.zeros_like(f_data[:1000,idx_i]), 5, color="r", alpha=0.8, vmin=1, marker="x")
    #ax.set_xlim(eval_grid.min(),eval_grid.max())
    #ax.set_ylim(eval_grid.min(),eval_grid.max())
    #fig.colorbar(c, ax=ax)
def plot_witness_loss(losses,init_data, final_data, net="G", method = "", num_final_it = 100,plot_witness=False):
    fig, ax = plt.subplots(1,4, figsize=(24,5))
    witness_val_init,norm_grad_witness_init,t_data_init, f_data_init, eval_grid_init = init_data
    witness_val,norm_grad_witness,t_data, f_data, eval_grid = final_data
    ax[0].plot(losses )
    plot_pdf(fig,ax[1],witness_val_init,t_data_init, f_data_init, eval_grid_init,plot_witness=plot_witness)
    plot_pdf(fig,ax[2],witness_val,t_data, f_data, eval_grid,plot_witness=plot_witness)
    plot_pdf(fig,ax[3],norm_grad_witness,t_data, f_data, eval_grid,plot_data=False,plot_witness=plot_witness)
    ax[0].set_xlabel(net + " iterations")
    ax[0].set_title( method+"loss per " + net + " iteration")
    ax[1].set_title("witness function: initial generator")
    ax[2].set_title("witness function:  generator at"+str(num_final_it)+ "iterations")
    ax[3].set_title("norm gradient witness function:  generator at" +str(num_final_it)+ "iterations")
def plot_witness_loss_1d(losses,init_data, final_data, net="G", method = "", num_final_it = 100, title="" ,plot_witness=False):
    fig, ax = plt.subplots(1,4, figsize=(24,5))
    st = fig.suptitle(title, fontsize="x-large")
    witness_val_init,norm_grad_witness_init,t_data_init, f_data_init, eval_grid_init = init_data
    witness_val,norm_grad_witness,t_data, f_data, eval_grid = final_data
    ax[0].plot(losses )
    plot_pdf_1d(fig,ax[1],witness_val_init,t_data_init, f_data_init, eval_grid_init,plot_witness=plot_witness)
    plot_pdf_1d(fig,ax[2],witness_val,t_data, f_data, eval_grid,plot_witness=plot_witness)
    plot_pdf_1d(fig,ax[3],norm_grad_witness,t_data, f_data, eval_grid,plot_data=False,plot_witness=plot_witness)
    ax[0].set_xlabel(net + " iterations")
    ax[0].set_title( method+"loss per " + net + " iteration")
    ax[1].set_title("witness function: initial generator")
    ax[2].set_title("witness function:  generator at"+str(num_final_it)+ "iterations")
    ax[3].set_title("norm gradient witness function:  generator at" +str(num_final_it)+ "iterations")

def plot_particles(axes,out):
    num_iteration = len(out["particles"])
    assert len(axes)==num_iteration+1
    for j in range(num_iteration):
        axes[j].scatter(out["particles"][j][:,0],out["particles"][j][:,1],color= "r")
        axes[0].set_title( str(out["particles_iter"][j]))
    axes[-1].scatter(out["target_particles"][:,0],out["target_particles"][:,1],color= "g")
    
