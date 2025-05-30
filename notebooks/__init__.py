# Add the commands below to the first cell of the notebook to have all necessary imports

# from notebooks.__init__ import *
# from notebooks.__init__ import *
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path
import sys

module_path = "/home/konstantin/personal/Summertime/"
if module_path not in sys.path:
    sys.path.append(module_path)

import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from types import SimpleNamespace
from hydra import compose, initialize_config_dir
from clearml import Task, InputModel

np.set_printoptions(suppress=True)

from source.hydra_config_wrapper import HydraConfigWrapper
from source.data_module import DataModule

from source.method.ness_detector import NeSSDetector
from source.method.syss.base_detectors.instantiate_base_detector import instantiate_base_detector

from source.method.syss.patch_detector import PatchDetector
from source.method.syss.synthetic_homography_distribution import SyntheticHomographyDistribution
from source.method.syss.warped_patch_generator import WarpedPatchGenerator
from source.method.syss.keypoint_projection_generator import KeypointProjectionGenerator
from source.method.syss.beta_eme import BetaEME

from source.evaluation.l2_matcher import L2Matcher
from source.evaluation.two_view_geometry import TwoViewGeometryEstimator, relative_pose_error_from_extrinsics, relative_pose_error, match_and_estimate_rel_pose_error_batch, relative_pose_accuracy

from source.visualization.two_view_geometry.test import RelPoseAccuracyPlotter, RelPoseMeanNumInlPlotter
from source.visualization.classical_metrics import ClassicalMetricsPlotter
from source.utils.visualization.plotting import plot_images, plot_keypoints, plot_matches
from source.utils.visualization.drawing import draw_patch_border, draw_keypoint

from source.utils.common_utils import gather_kp
from source.utils.method import instantiate_method, get_features
from source.utils.io import get_inference_filename, save_features, save_figure
from source.utils.visualization.miscellaneous import get_baselines_plot_group, sort_by_aliases, get_random_hex_color, get_ness_st_beta_ablation_plot_group, get_boness_st_beta_ablation_plot_group

from source.visualization.painters.keypoint_patch_painter import KeypointPatchPainter
from source.visualization.painters.keypoint_projection_painter import KeypointProjectionPainter
from source.visualization.painters.warped_visualization_patch_painter import WarpedVisualizationPatchPainter

