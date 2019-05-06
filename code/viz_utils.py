
import matplotlib.pyplot as plt
from Utils import *
import os
import pandas as pd

import tensorflow as tf



def save(n, dirname,ext='.pdf', save_figs=True, **kwargs):
    if save_figs==True:
        kwargs.setdefault('bbox_inches', 'tight')
        kwargs.setdefault('pad_inches', 0)
        kwargs.setdefault('transparent', True)
        plt.savefig(os.path.join(dirname, n + ext), **kwargs)
def set_lim(xlims = None, ylims= None):
    if not xlims is None:
        plt.xlim(xlims[0], xlims[1])
    if not ylims is None:
        plt.ylim(ylims[0], ylims[1])
    #plt.axis('off')

def fig(xlims = None, ylims = None, num_subplots = None):
    x_size = 6
    y_size = 4
    if num_subplots is None:
        f = plt.figure(figsize=(x_size, y_size),constrained_layout=True)
        ax = None
    else:
        f,ax = plt.subplots(num_subplots[0], num_subplots[1],figsize=(num_subplots[1]*x_size, num_subplots[0]*y_size),constrained_layout=True)
    set_lim(xlims, ylims)
    return f, ax

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
    ax[0].set_yscale("log", nonposy='clip')
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
    ax[0].set_yscale("log", nonposy='clip')

def plot_particles(axes,out):
    num_iteration = len(out["particles"])
    assert len(axes)==num_iteration+1
    for j in range(num_iteration):
        axes[j].scatter(out["particles"][j][:,0],out["particles"][j][:,1],color= "r")
        axes[0].set_title( str(out["particles_iter"][j]))
    axes[-1].scatter(out["target_particles"][:,0],out["target_particles"][:,1],color= "g")


def last_run(df, step):
    d_first = df.loc[df['step'] == step]
    ind = 0
    if not d_first.empty: 
        ind = d_first.index[-1]
    return df[ind:]

def plot_scalar(ax,all_events,selected_runs,scalar_name,scalar_key,linewidth = 2.,xlabel='',ylabel='',title='',colors=None, xlim=None, ylim=None, xscale='linear',yscale='linear',no_legend=False, step=1):
    if colors is None:
        colors = sns.color_palette("Set1", n_colors=len(selected_runs), desat=.7)
    for iter_num, key_values in enumerate(selected_runs.items()):
        key, value = key_values
        run_name = os.path.join(value,scalar_name,scalar_key)
        scalars = all_events.Scalars(run_name,scalar_name)
        df = last_run(pd.DataFrame(scalars), step)

        df.plot(x='step',y='value',ax =ax, linewidth=linewidth, label=key,color =colors[iter_num])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    if no_legend:
        ax.get_legend().remove()



def plot_particles(ax,selected_runs,ckpt_values,linewidth = 2.,xlabel='',ylabel='',title='',colors=None, xlim=None, ylim=None, xscale='linear',yscale='linear',no_legend=False, step=1):

    for iter_num, key_values in enumerate(selected_runs.items()):
        key, value = key_values
        run_name = os.path.join(value,'checkpoint','ckpt.iter_'+str(ckpt_values[iter_num]))
        reader = tf.train.NewCheckpointReader(ckpt_fpath)

        param_map = reader.get_variable_to_shape_map()

    for k, v in param_map.items():
        ax.scatter(v[0,:], v[1,:],label=key,color =colors[iter_num])


