"""plot.py"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2D_image(z_net, z_gt, step, var=5):
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)

    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))

    ax1.set_title('Thermodynamics-informed GNN')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error')
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    # Adjust ranges
    X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[step, :, var].min(), z_gt[step, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    # Initial snapshot
    q1_net, q3_net = z_net[step, :, 0], z_net[step, :, 1]
    q1_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 1]
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    fig.savefig(os.path.join('outputs/images/', f'beam_{step}.png'))

    # Oculta las marcas de los ejes y las etiquetas
    ax1.tick_params(axis='both', which='both', length=0)
    plt.savefig("grafico.svg", format="svg")


def plot_2D(z_net, z_gt, output_dir, var=5):
    T = z_net.size(0)
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Adjust ranges
    X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())

    # Initial snapshot
    q1_net, q3_net = z_net[0, :, 0], z_net[0, :, 1]
    q1_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    # e = var_net.numpy() - var_gt.numpy()
    # gt = var_gt.numpy()
    # var_error = ((e ** 2) / (gt ** 2)) ** 0.5
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
        # Bounding box
        for xb, yb in zip(Xb, Yb):
            ax1.plot([xb], [yb], 'w')
            ax2.plot([xb], [yb], 'w')
            ax3.plot([xb], [yb], 'w')
        # Scatter points
        q1_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 1]
        q1_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 1]
        # var_net = calculateBorders(z_net[snap, :, :3], h, r1, r2)
        # var_gt = calculateBorders(z_gt[snap, :, :3], h, r1, r2)
        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net
        # e = var_net.numpy() - var_gt.numpy()
        # gt = var_gt.numpy()+0.000000000001
        # var_error = ((e) / (gt))
        # var_gt = var_gt*0
        # var_gt = z_net[snap, :, 5]
        # ax1.set(xlim=(-0.04, 0.04), ylim=(-0.04, 0.04), zlim=(-0.01, 0.08))
        ax1.scatter(q1_net, q3_net, c=var_net,
                    vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                    vmin=z_min)
        ax3.scatter(q1_net, q3_net, c=var_error,
                    vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join('images/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    save_dir = os.path.join(output_dir, 'beam.gif')
    anim.save(save_dir, writer=writergif)
