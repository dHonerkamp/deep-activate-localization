#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Wedge
from scipy.special import softmax

def draw_text(text, bgcolor, plt_ax, text_plt):
    """
    Render the text
    :param str text: text to render
    :param str bgcolor: backgroundcolor used to render text
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :param matplotlib.text.Text text_plt: plot of text
    :return matplotlib.text.Text: updated plot of text
    """

    if text_plt is None:
        # render text with color
        text_plt = plt_ax.text(0.97, 0.97, text, backgroundcolor=bgcolor,
                        horizontalalignment='right', verticalalignment='top',
                        transform=plt_ax.transAxes, fontsize=12)
    else:
        # update existing text
        text_plt.set_text(text)

    return text_plt

def draw_floor_map(floor_map, map_shape, plt_ax, map_plt, cmap='gray'):
    """
    Render the scene floor map
    :param ndarray floor_map: environment scene floor map
    :param ndarray map_shape: (unpadded) map shape [H, W, C]
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :return matplotlib.image.AxesImage: updated plot of scene floor map
    """

    H, W = map_shape[:2]
    origin_x, origin_y = W / 2, H / 2
    if map_plt is None:
        # draw floor map
        # map_plt = plt_ax.imshow(floor_map[:H, :W], cmap=cmap, origin='lower')
        map_plt = plt_ax.imshow(floor_map[:H, :W], cmap=cmap)
        # plt.scatter(origin_x, origin_y, s=10, c='black', marker='x', alpha=1)

        # rect = patches.Rectangle((45, 45), 10, 10, linewidth=1, edgecolor='white', facecolor='none')
        # plt_ax.add_patch(rect)
    else:
        map_plt.set_data(floor_map[:H, :W])
    return map_plt

def draw_particles_pose(particles, weights, map_shape, particles_plt, scale=1):
    """
    Render the particle poses on the scene floor map
    :param ndarray particles: estimates of particle pose
    :param ndarray weights: corresponding weights of particle pose estimates
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param matplotlib.collections.PathCollection: plot of particle position color coded according to weights
    :param scale: integer rescaling value
    :return matplotlib.collections.PathCollection: updated plot of particles
    """

    part_col, part_row, part_th = tf.unstack(particles, axis=-1, num=3)   # (k, 3)
    # height, width, channel = map_shape

    # HACK: display particles alpha proprtional to their weights
    lin_weights = softmax(weights)
    th = 0 # np.mean(lin_weights)
    alphas = np.where(lin_weights >= th, 1, 0) * lin_weights
    alphas = alphas/np.max(alphas)

    rgba_colors = cm.rainbow(weights - np.min(weights))
    rgba_colors[:, 3] = alphas

    if particles_plt is None:
        # render particles positions with color
        particles_plt = plt.scatter(part_row, part_col, s=10, c=rgba_colors)
    else:
        # update existing particles positions and color
        particles_plt.set_offsets(tf.stack([part_row, part_col], axis=-1))
        particles_plt.set_color(rgba_colors)

    return particles_plt

def draw_robot_pose(robot_pose, color, map_shape, plt_ax, position_plt, heading_plt, plt_path=False, scale=1):
    """
    Render the robot pose on the scene floor map
    :param ndarray robot_pose: ndarray representing robot position (x, y) and heading (theta)
    :param str color: color used to render robot position and heading
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :param matplotlib.patches.Wedge position_plt: plot of robot position
    :param matplotlib.lines.Line2D heading_plt: plot of robot heading
    :param scale: integer rescaling value
    :return tuple(matplotlib.patches.Wedge, matplotlib.lines.Line2D): updated position and heading plot of robot
    """

    # col, row, heading = np.squeeze(robot_pose)
    # NOTE: matplotlib expects x, y coords, while when we index into the floormap we index [y, x]
    y, x, heading = np.squeeze(robot_pose)
    # height, width, channel = map_shape

    heading_len = robot_radius = 3 * 1.0
    xdata = [x, x + (robot_radius + heading_len) * np.cos(heading)]
    ydata = [y, y + (robot_radius + heading_len) * np.sin(heading)]

    if position_plt == None:
        # render robot position and heading with color
        position_plt = Wedge((x, y), robot_radius, 0, 360, color=color, alpha=0.5)
        plt_ax.add_artist(position_plt)
        heading_plt = plt_ax.plot(xdata, ydata, color=color, alpha=0.5)[0]
    else:
        # update existing robot position and heading
        position_plt.update({'center': [x, y]})
        heading_plt.update({'xdata': xdata, 'ydata': ydata})
    if plt_path is True:
        plt_ax.arrow(xdata[0], ydata[0], (xdata[1]-xdata[0]), (ydata[1]-ydata[0]), head_width=0.5, head_length=0.7, fc='brown', ec='brown')

    return position_plt, heading_plt
