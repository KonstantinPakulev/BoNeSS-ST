import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from source.utils.visualization.drawing import draw_image, draw_cv_keypoints
from source.utils.visualization.miscellaneous import if_torch2cv, torch2cv, to_cv_keypoint, if_torch2numpy


def plot_images(image, nrows=1, ncols=1,
                figsize=(18, 18), font_size=None,
                return_axes=False, show_titles=True):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])

    if isinstance(image, dict):
        for axis, key in zip(axes.ravel(), image.keys()):
            draw_image(axis, if_torch2cv(image[key]), title=key if show_titles else None, fontsize=font_size)
        
    elif isinstance(image, list):
        for axis, _image in zip(axes.ravel(), image):
            draw_image(axis, if_torch2cv(_image))
        
    else:
        draw_image(axes[0], if_torch2cv(image))
    
    if return_axes:
        return fig, axes


def plot_keypoints(image, 
                   kp, kp_mask=None,
                   nrows=1, ncols=1,
                   figsize=(18, 18)):
    
    if isinstance(image, dict):
        raise NotImplementedError("Plotting a dictionary of images is not implemented yet.")
    
    elif isinstance(image, list):
        if kp_mask is None:
            kp_mask = [None] * len(kp)

        else:
            kp_mask = [_kp_mask.cpu().numpy() for _kp_mask in kp_mask]

        plot_images([draw_cv_keypoints(torch2cv(_image), _kp.cpu().numpy(), _kp_mask)
                     for _image, _kp, _kp_mask in zip(image, kp, kp_mask)],
                    nrows=nrows, ncols=ncols,
                    figsize=figsize)
        
    else:
        plot_images(draw_cv_keypoints(torch2cv(image), kp.cpu().numpy(), kp_mask), 
                    figsize=figsize)


def plot_matches(image1, image2, kp1, nn_kp2, mask1=None, figsize=(30, 15)):
    image1, image2 = torch2cv(image1), torch2cv(image2)

    kp1, nn_kp2 = kp1.cpu().numpy(), nn_kp2.cpu().numpy()

    if mask1 is not None:
        mask1 = if_torch2numpy(mask1)
    
    cv_kp1, cv_nn_kp2 = to_cv_keypoint(kp1), to_cv_keypoint(nn_kp2)

    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1)) if mask1 is None or mask1[i]]

    img_matches = cv2.drawMatches(image1, cv_kp1, image2, cv_nn_kp2, matches, None,
                                 matchColor=(0, 255, 0),
                                 singlePointColor=(255, 0, 0),
                                 matchesMask=None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    _, ax = plt.subplots(figsize=figsize)

    ax.imshow(img_matches)
    ax.set_axis_off()
