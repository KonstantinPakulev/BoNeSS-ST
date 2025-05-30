import cv2
import numpy as np

from matplotlib.patches import Rectangle, Ellipse, Circle
from matplotlib.lines import Line2D
from matplotlib import transforms
from numbers import Number

from source.utils.visualization.miscellaneous import to_cv_keypoint


def draw_image(axis,
               image: np.ndarray,
               title=None, **title_kwargs):
    axis.imshow(image, cmap='gray')

    if title is not None:
        axis.set_title(title, title_kwargs)
    
    axis.set_axis_off()


def draw_patch_border(axis, patch_center, rect_size, **border_kwargs):
    center = patch_center - rect_size / 2

    draw_rectangle(axis, center, rect_size, fc='none', **border_kwargs)


def draw_keypoint(axis, kp, **marker_kwargs):
    draw_marker(axis, kp,
                marker='X', 
                markerfacecolor='darkorange', 
                markeredgecolor='black',
                **marker_kwargs)


def draw_keypoint_measurement(axis, kp_measurement, color='red', **marker_kwargs):
    draw_marker(axis, kp_measurement, marker='x', color=color, **marker_kwargs)


def draw_keypoint_projection_distribution_mean(axis, kps_projs_mean, color='yellow', **marker_kwargs):
    draw_marker(axis, kps_projs_mean, marker='+', color=color, **marker_kwargs)


def draw_keypoint_projection_distribution_covariance(axis, kps_projs_dist_mean, kps_projs_dist_cov, color='yellow',
                                                    **ellipse_kwargs):
    draw_covariance(axis,
                    kps_projs_dist_mean, kps_projs_dist_cov,
                    ec=color, ls='-', zorder=2, **ellipse_kwargs)


def draw_delta(axis, kp, kps_projs_dist_mean):
    draw_line(axis, kp, kps_projs_dist_mean, color='moccasin', lw=1, ls='--', dashes=(2,1))


"""
General drawing functions
"""
    

def draw_marker(axis, position, **marker_kwargs):
    position = input2plot_point(position)

    axis.plot(position[0], position[1], **marker_kwargs)


def draw_rectangle(axis, position, size, **rectangle_kwargs):
    position = input2plot_point(position)

    rect = Rectangle(position, size, size, **rectangle_kwargs)
    axis.add_patch(rect)


def draw_covariance(axis, mean, covariance, **ellipse_kwargs):
    pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    ell_radius_x = np.sqrt(1 - pearson)
    ell_radius_y = np.sqrt(1 + pearson)

    ellipse = Ellipse((0, 0), 
                      width=ell_radius_y * 3, 
                      height=ell_radius_x * 3, 
                      fc='none', 
                      **ellipse_kwargs)
    
    scale_x = np.sqrt(covariance[0, 0])
    scale_y = np.sqrt(covariance[1, 1])

    transformation = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_y, scale_x) \
        .translate(mean[1], mean[0])

    ellipse.set_transform(transformation + axis.transData)

    axis.add_patch(ellipse)


def draw_line(axis, point1, point2, **kwargs):
    point1 = input2plot_point(point1)
    point2 = input2plot_point(point2)
    
    line = Line2D((point1[0], point2[0]), (point1[1], point2[1]), **kwargs)
    axis.add_line(line)


def draw_circle(axis, position, radius, **kwargs):
    position = input2plot_point(position)
    
    circle = Circle(position, radius, **kwargs)
    axis.add_patch(circle)


"""
Matplotlib functions
"""

def autolabel(ax, rects, label=None, fontsize=None, precision=3):
    for rect in rects:
        height = rect.get_height()

        ax.annotate(('{:.' + str(precision) + 'f}').format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)

        if label is not None:
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=fontsize)


"""
Support functions
"""

def input2plot_point(input):
     if isinstance(input, Number):
        return np.array([input, input])
     
     else:
         return input[[1, 0]]
     

"""
OpenCV functions
"""


def draw_cv_keypoints(image: np.ndarray, kp: np.ndarray,
                      draw_mask: np.ndarray = None,
                      color=(0, 255, 0)):
    """
    :param image: C x H x W
    :param kp: N x 2
    :param draw_mask: N
    :param color: tuple (r, g, b)
    """
    if draw_mask is not None:
        kp = kp[draw_mask]

    cv_kp = to_cv_keypoint(kp)
    return cv2.drawKeypoints(image, cv_kp, None, color=color)
